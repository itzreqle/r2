"""Intent catalog and capability flags."""

from __future__ import annotations

from narrow_ai.intents.schemas import IntentName, IntentSpec, PermissionLevel


class IntentRegistry:
    def __init__(self, specs: list[IntentSpec]) -> None:
        self._by_name = {s.name: s for s in specs}

    def get(self, name: IntentName) -> IntentSpec:
        return self._by_name[name]

    def all(self) -> list[IntentSpec]:
        return list(self._by_name.values())

    def implemented(self) -> list[IntentSpec]:
        return [s for s in self.all() if s.implemented]

    def help_text(self) -> str:
        lines = ["Supported capabilities:"]
        for spec in self.implemented():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines)


def default_registry() -> IntentRegistry:
    """Phase 1: only general_help, git_status, search_files, clarify, unsupported."""
    specs = [
        IntentSpec(
            name=IntentName.GENERAL_HELP,
            description="Explain supported functions",
            permission=PermissionLevel.NONE,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.GIT_STATUS,
            description="Report branch and working-tree status",
            permission=PermissionLevel.READ,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.GIT_DIFF_SUMMARY,
            description="Summarize git changes (read-only)",
            permission=PermissionLevel.READ,
            implemented=False,
        ),
        IntentSpec(
            name=IntentName.SEARCH_FILES,
            description="Search filenames in the workspace",
            permission=PermissionLevel.READ,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.READ_FILE,
            description="Read an approved workspace file",
            permission=PermissionLevel.READ,
            implemented=False,
        ),
        IntentSpec(
            name=IntentName.EXPLAIN_TEST_OUTPUT,
            description="Explain supplied test output",
            permission=PermissionLevel.NONE,
            implemented=False,
        ),
        IntentSpec(
            name=IntentName.RUN_TESTS,
            description="Run an allowlisted test command",
            permission=PermissionLevel.EXECUTE,
            implemented=False,
        ),
        IntentSpec(
            name=IntentName.OPEN_APP,
            description="Open an approved application",
            permission=PermissionLevel.EXECUTE,
            implemented=False,
        ),
        IntentSpec(
            name=IntentName.REMEMBER_PREFERENCE,
            description="Store a preference after explicit agreement",
            permission=PermissionLevel.WRITE,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.RECALL_PREFERENCE,
            description="Retrieve a stored preference",
            permission=PermissionLevel.READ,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.CLARIFY_REQUEST,
            description="Ask a follow-up when ambiguous",
            permission=PermissionLevel.NONE,
            implemented=True,
        ),
        IntentSpec(
            name=IntentName.UNSUPPORTED,
            description="Decline out-of-domain requests",
            permission=PermissionLevel.NONE,
            implemented=True,
        ),
    ]
    return IntentRegistry(specs)
