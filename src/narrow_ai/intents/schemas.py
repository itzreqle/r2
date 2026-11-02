"""Typed intent contracts. Model output must validate against these."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class IntentName(StrEnum):
    GENERAL_HELP = "general_help"
    GIT_STATUS = "git_status"
    GIT_DIFF_SUMMARY = "git_diff_summary"
    SEARCH_FILES = "search_files"
    READ_FILE = "read_file"
    EXPLAIN_TEST_OUTPUT = "explain_test_output"
    RUN_TESTS = "run_tests"
    OPEN_APP = "open_app"
    REMEMBER_PREFERENCE = "remember_preference"
    RECALL_PREFERENCE = "recall_preference"
    ANSWER_QUESTION = "answer_question"
    CLARIFY_REQUEST = "clarify_request"
    UNSUPPORTED = "unsupported"


class PermissionLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NONE = "none"


class EmptyParams(BaseModel):
    model_config = {"extra": "forbid"}


class SearchFilesParams(BaseModel):
    model_config = {"extra": "forbid"}

    pattern: str = Field(..., min_length=1, max_length=256)
    workspace: Literal["active"] = "active"

    @field_validator("pattern")
    @classmethod
    def no_shell_metachar(cls, value: str) -> str:
        banned = [";", "|", "&", "`", "$("]
        if any(b in value for b in banned):
            raise ValueError("pattern contains forbidden characters")
        return value


class ReadFileParams(BaseModel):
    model_config = {"extra": "forbid"}

    path: str = Field(..., min_length=1, max_length=1024)


class ExplainTestOutputParams(BaseModel):
    model_config = {"extra": "forbid"}

    # Empty = explain the output of the last run_tests in this session.
    output: str = Field(default="", max_length=50_000)


class RunTestsParams(BaseModel):
    model_config = {"extra": "forbid"}

    # An id from tools.test_commands; empty = the first configured command. The model
    # only ever picks an id: the argv comes from config, never from model text.
    command_id: str = Field(default="", max_length=64)


class OpenAppParams(BaseModel):
    model_config = {"extra": "forbid"}

    app_name: str = Field(..., min_length=1, max_length=128)


class RememberPreferenceParams(BaseModel):
    model_config = {"extra": "forbid"}

    key: str = Field(..., min_length=1, max_length=64)
    value: str = Field(..., min_length=1, max_length=1024)


class RecallPreferenceParams(BaseModel):
    model_config = {"extra": "forbid"}

    key: str = Field(..., min_length=1, max_length=64)


class AnswerQuestionParams(BaseModel):
    model_config = {"extra": "forbid"}

    question: str = Field(..., min_length=1, max_length=1000)


class ClarifyRequestParams(BaseModel):
    model_config = {"extra": "forbid"}

    question: str = Field(..., min_length=1, max_length=512)


class UnsupportedParams(BaseModel):
    model_config = {"extra": "forbid"}

    reason: str = Field(default="Outside supported domain", max_length=512)


class GeneralHelpParams(BaseModel):
    model_config = {"extra": "forbid"}

    topic: str = Field(default="", max_length=256)


PARAMS_BY_INTENT: dict[IntentName, type[BaseModel]] = {
    IntentName.GENERAL_HELP: GeneralHelpParams,
    IntentName.GIT_STATUS: EmptyParams,
    IntentName.GIT_DIFF_SUMMARY: EmptyParams,
    IntentName.SEARCH_FILES: SearchFilesParams,
    IntentName.READ_FILE: ReadFileParams,
    IntentName.EXPLAIN_TEST_OUTPUT: ExplainTestOutputParams,
    IntentName.RUN_TESTS: RunTestsParams,
    IntentName.OPEN_APP: OpenAppParams,
    IntentName.REMEMBER_PREFERENCE: RememberPreferenceParams,
    IntentName.RECALL_PREFERENCE: RecallPreferenceParams,
    IntentName.ANSWER_QUESTION: AnswerQuestionParams,
    IntentName.CLARIFY_REQUEST: ClarifyRequestParams,
    IntentName.UNSUPPORTED: UnsupportedParams,
}


class IntentDecision(BaseModel):
    """Authoritative structured decision after validation."""

    model_config = {"extra": "forbid"}

    intent: IntentName
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False

    def validated_params(self) -> BaseModel:
        model = PARAMS_BY_INTENT[self.intent]
        return model.model_validate(self.parameters)


class IntentSpec(BaseModel):
    name: IntentName
    description: str
    permission: PermissionLevel
    timeout_sec: int = 30
    implemented: bool = False
