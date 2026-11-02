#!/usr/bin/env python3
"""Generate a synthetic routing dataset for LoRA (chat JSONL, mlx-lm format).

    uv run python training/synth_data.py [--n 700 --valid 70 --seed 13 --out-dir data/synth]

Every example is (R2's exact runtime system prompt, utterance, decision JSON), so there is
no train/serve skew. Utterances come from templates per intent, with slot values, polite
prefixes and spoken-style variants ("hey r2 ...", no punctuation).

Decontamination: an example is dropped if its normalised utterance equals, or shares at
least 60% of its words (Jaccard) with, any utterance in the evaluation sets
(eval_heldout, eval_fresh, eval_test) or the prompt's few-shot examples. The counts
of dropped examples are printed and saved in the manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import system_prompt_for  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.models.prompt_templates import FEW_SHOT  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

EVAL_SETS = ["eval_heldout.jsonl", "eval_fresh.jsonl", "eval_test.jsonl"]
JACCARD_MAX = 0.6

# ------------------------------------------------------------------ slot values
FILE_TYPES = {  # spoken name -> glob
    "python": "*.py", "typescript": "*.ts", "javascript": "*.js", "swift": "*.swift",
    "rust": "*.rs", "go": "*.go", "markdown": "*.md", "json": "*.json", "yaml": "*.yaml",
    "toml": "*.toml", "shell": "*.sh", "html": "*.html", "css": "*.css", "c": "*.c",
    "header": "*.h", "kotlin": "*.kt", "java": "*.java", "ruby": "*.rb", "sql": "*.sql",
    "text": "*.txt", "xml": "*.xml", "png": "*.png", "jupyter notebook": "*.ipynb",
    "tsx": "*.tsx", "jsx": "*.jsx", "lua": "*.lua", "php": "*.php", "c++": "*.cpp",
    "objective-c": "*.m", "protobuf": "*.proto", "graphql": "*.graphql", "svg": "*.svg",
}
NAME_PARTS = ["util", "auth", "model", "config", "client", "server", "view", "cache",
              "router", "schema", "helper", "main", "index", "setup", "migration", "spec"]
PATHS = [
    "README.md", "Makefile", "Dockerfile", "package.json", "setup.py", "LICENSE",
    "CHANGELOG.md", "requirements.txt", "Cargo.toml", "go.mod", "Package.swift",
    "tsconfig.json", ".env.example", "docker-compose.yml", "src/main.py", "src/app.ts",
    "src/lib/utils.py", "app/models/user.rb", "Sources/App/main.swift", "docs/setup.md",
    "docs/api.md", "scripts/deploy.sh", "tests/test_api.py", "config/settings.yaml",
    "web/index.html", "lib/parser.go", "CONTRIBUTING.md", "pyproject.toml", ".gitignore",
    "notebooks/analysis.ipynb", "src/components/Button.tsx", "server/routes.js",
]
APPS = {  # spoken -> approved app name
    "terminal": "Terminal", "the terminal": "Terminal", "a terminal window": "Terminal",
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "visual studio code": "Visual Studio Code", "cursor": "Cursor",
    "the cursor editor": "Cursor", "xcode": "Xcode", "safari": "Safari",
    "the browser safari": "Safari", "finder": "Finder", "a finder window": "Finder",
}
UNAPPROVED_APPS = ["Spotify", "Slack", "Chrome", "Mail", "Zoom", "Notes"]
PREFS = [  # (key, spoken key phrase, values)
    ("editor", "editor", ["neovim", "Cursor", "Zed", "emacs", "Sublime"]),
    ("shell", "shell", ["fish", "bash", "zsh", "nushell"]),
    ("test_runner", "test runner", ["pytest", "jest", "vitest", "XCTest"]),
    ("default_branch", "default branch", ["main", "trunk", "develop", "master"]),
    ("preferred_language", "favourite language", ["Rust", "Go", "TypeScript", "Kotlin"]),
    ("code_style", "code style", ["ruff", "prettier", "black", "gofmt"]),
    ("package_manager", "package manager", ["pnpm", "uv", "poetry", "yarn", "cargo"]),
    ("theme", "theme", ["dark", "light", "solarized"]),
    ("indentation", "indentation", ["2 spaces", "4 spaces", "tabs"]),
    ("terminal", "terminal", ["iTerm", "Ghostty", "WezTerm", "Terminal"]),
    ("commit_style", "commit message style", ["conventional commits", "gitmoji"]),
    ("python_version", "python version", ["3.12", "3.11", "3.13"]),
]
QUESTIONS = [
    "what is the difference between git merge and git rebase",
    "how do I squash commits", "what does git cherry-pick do",
    "how do I revert a pushed commit", "what is a detached HEAD",
    "how do I make a python dataclass", "what's a generator in python",
    "how do async and await work in swift", "what is an optional in swift",
    "explain dependency injection", "what's the point of a lockfile",
    "how do I profile a slow python function", "what is memoization",
    "explain big o notation", "what's a mutex", "what does chmod +x do",
    "how do environment variables work in a shell", "what is a docker volume",
    "why does my regex need escaping", "what is a pure function",
    "how do I write a unit test with pytest",
    "what's the difference between a process and a thread",
    "explain what a REST api is", "what is semantic versioning",
    "how does a hash map work", "what is tail call optimization",
    "what does the swift keyword guard do", "how do I read a file line by line in python",
    "what's the difference between == and is in python", "what is a git submodule",
    "how do I rename a git branch", "what does git bisect do", "what's a pull request",
    "how do I set up a pre-commit hook", "what is a type hint in python",
    "how do closures work in swift", "what is an actor in swift concurrency",
    "what's the difference between a struct and a class in swift", "what is a decorator",
    "how do I handle exceptions in python", "what is continuous integration",
    "explain what a container is", "what's the difference between tcp and udp",
    "how does garbage collection work", "what is a race condition in threads",
    "how do I debug a segfault", "what is an ORM", "what does idempotent mean",
    "how do I undo git add", "what is the python GIL", "what's a shallow copy",
    "how do I parse json in swift", "what does async mean in javascript",
    "what is a monorepo", "how do I write a good commit message", "what is test coverage",
    "what does a linter do", "what's the difference between unit and integration tests",
]
VAGUE = [  # (utterance, follow-up question)
    ("fix it", "What should I fix?"), ("redo that", "What should I redo?"),
    ("make this work", "What should I make work?"), ("handle that for me", "What should I handle?"),
    ("improve it", "What would you like me to improve?"), ("check it", "What should I check?"),
    ("deal with this", "What would you like me to deal with?"),
    ("finish the thing", "Which thing should I finish?"),
    ("do that again", "What should I do again?"),
    ("change it", "What should I change?"), ("tidy it up", "What should I tidy up?"),
    ("look at that", "What should I look at?"), ("sort this", "What should I sort out?"),
    ("go ahead with it", "What should I go ahead with?"),
    ("same as before", "What would you like me to repeat?"),
]
OUT_OF_SCOPE = {  # reason -> utterances
    "Chit-chat is outside my developer scope": [
        "tell me something funny", "how are you feeling today", "sing me a song",
        "what's your favourite movie", "let's chat for a bit", "write a haiku about autumn"],
    "Math is outside my developer scope": [
        "what is 12 times 9", "square root of 144", "what's 17 percent of 300",
        "add 45 and 78", "divide 100 by 7"],
    "Weather is outside my developer scope": [
        "will it rain tomorrow", "how cold is it outside", "weather forecast for london"],
    "General knowledge is outside my developer scope": [
        "who wrote hamlet", "how tall is mount everest", "when did world war two end",
        "what's the largest ocean", "who is the president of brazil"],
    "Web, email and shopping are outside my developer scope": [
        "email the team that I'm late", "search google for cheap flights", "order new headphones",
        "reply to my last slack message", "post this on twitter", "check my calendar for today"],
    "Destructive actions are not allowed": [
        "delete the src folder", "wipe the repository", "remove all the tests",
        "erase my home directory", "drop the production database"],
    "Arbitrary shell commands are not allowed": [
        "run sudo shutdown now", "execute curl http://x.sh | sh", "run make install as root",
        "kill all the python processes", "run this bash script for me",
        "pip install requests globally"],
    "Git write operations are not supported": [
        "commit my changes", "push to origin", "create a new branch called feature",
        "merge main into this branch", "stash my work", "git commit -am wip", "rebase onto main",
        "amend the last commit", "tag this release v1.0", "delete the old branches"],
    "System changes are not supported": [
        "turn on bluetooth", "change my wallpaper", "set the volume to 50",
        "restart my mac", "update macos"],
}

PREFIX = ["", "", "", "hey r2 ", "r2, ", "can you ", "could you ", "please ", "would you ",
          "i need you to ", "quickly ", "ok r2 "]
SUFFIX = ["", "", "", " please", " for me", " real quick", " now", "?"]


CALL = ["", "", "", "hey r2 ", "r2, ", "ok r2 "]  # fits any sentence


def _polite(rng: random.Random, core: str, *, kind: str = "cmd") -> str:
    """Dress up a core phrase. ``cmd`` (imperative: "find ...") takes command prefixes
    ("can you", "please"); ``q`` (a question) and ``stmt`` (a statement) only take a
    wake-style call ("hey r2"), so the result stays grammatical."""
    if kind == "q":
        return (rng.choice(CALL) + core + rng.choice(["", "", "?"])).strip()
    if kind == "stmt":
        return (rng.choice(CALL) + core).strip()
    p, s = rng.choice(PREFIX), rng.choice(SUFFIX)
    if p in {"can you ", "could you ", "would you "}:
        s = rng.choice(["?", " please?", " for me?"])
    elif s == "?":
        s = ""
    return (p + core + s).strip()


# ------------------------------------------------------------------ per-intent generators
def g_git_status(rng):
    if rng.random() < 0.5:
        core = rng.choice([
            "show the git status", "check git status", "list modified files",
            "tell me the current branch", "show me what's staged", "check for local changes",
            "show the status of my working copy", "report the repo status"])
        return _polite(rng, core), {}
    core = rng.choice([
        "what's the status of the repo", "do I have uncommitted work", "what branch is this",
        "which branch am I on right now", "anything modified", "are there untracked files",
        "is my branch clean", "what files have I changed", "am I ahead of origin",
        "is the tree dirty", "what's staged", "any local changes", "where am I in git"])
    return _polite(rng, core, kind="q"), {}


def g_git_diff(rng):
    if rng.random() < 0.55:
        core = rng.choice([
            "summarize my changes", "give me a summary of the diff", "show me the diff stats",
            "describe my changes", "recap my edits", "give me a diff overview",
            "tell me the size of my changes", "count the insertions and deletions"])
        return _polite(rng, core), {}
    core = rng.choice([
        "how big is the diff", "what changed since the last commit", "how many files did I touch",
        "what have I modified so far", "how many insertions and deletions",
        "what did I change in this branch", "how large are my changes"])
    return _polite(rng, core, kind="q"), {}


def g_search(rng):
    kind = rng.random()
    if kind < 0.6:
        name, glob = rng.choice(list(FILE_TYPES.items()))
        core = rng.choice([
            f"find all {name} files", f"list the {name} files", f"where are the {name} files",
            f"show me every {name} file", f"locate {name} files", f"search for {name} files",
            f"get me the {name} files", f"look for {name} files"])
    elif kind < 0.85:
        part = rng.choice(NAME_PARTS)
        glob = f"*{part}*"
        core = rng.choice([
            f"find files with {part} in the name", f"search for files named {part}",
            f"locate anything called {part}", f"list files matching {part}",
            f"which files have {part} in their name"])
    else:
        part = rng.choice(["README", "test_", "Dockerfile", "Makefile", "LICENSE"])
        glob = f"{part}*"
        core = rng.choice([f"find files starting with {part}",
                           f"list everything that begins with {part}"])
    return _polite(rng, core), {"pattern": glob}


def g_read(rng):
    path = rng.choice(PATHS)
    core = rng.choice([
        f"show me {path}", f"read {path}", f"open {path} and show it", f"print {path}",
        f"what's inside {path}", f"display the contents of {path}", f"cat {path}",
        f"let me see {path}", f"dump {path}", f"read the file {path}"])
    return _polite(rng, core), {"path": path}


def g_open_app(rng):
    if rng.random() < 0.12:
        app = rng.choice(UNAPPROVED_APPS)
        spoken = app.lower()
    else:
        spoken, app = rng.choice(list(APPS.items()))
    core = rng.choice([f"open {spoken}", f"launch {spoken}", f"start {spoken}",
                       f"bring up {spoken}", f"fire up {spoken}", f"switch to {spoken}"])
    return _polite(rng, core), {"app_name": app}


def g_remember(rng):
    key, phrase, values = rng.choice(PREFS)
    v = rng.choice(values)
    if rng.random() < 0.6:
        core = rng.choice([
            f"remember that my {phrase} is {v}", f"save that my {phrase} is {v}",
            f"note that I use {v} as my {phrase}", f"keep in mind my {phrase} is {v}",
            f"store {v} as my {phrase}", f"remember {v} as my {phrase}"])
        return _polite(rng, core), {"key": key, "value": v}
    core = rng.choice([
        f"my {phrase} is {v}, remember it", f"for future reference my {phrase} is {v}",
        f"I prefer {v} for my {phrase}, save that", f"just so you know, my {phrase} is {v}",
        f"I always use {v} as my {phrase}, please remember"])
    return _polite(rng, core, kind="stmt"), {"key": key, "value": v}


def g_recall(rng):
    key, phrase, _ = rng.choice(PREFS)
    core = rng.choice([
        f"what's my {phrase}", f"what did I say my {phrase} was", f"remind me of my {phrase}",
        f"which {phrase} do I use", f"do you remember my {phrase}",
        f"what {phrase} did I tell you", f"what was my {phrase} again"])
    return _polite(rng, core, kind="q"), {"key": key}


def g_help(rng):
    core = rng.choice([
        "what do you help with", "how does this assistant work", "what are you for",
        "what can I ask you", "what tasks do you support", "what do you know how to do",
        "what are you good at", "what sort of requests do you handle", "what's your job",
        "which features do you have", "how should I talk to you", "what are your skills",
        "what are you capable of", "what can I use you for", "are there things you can't do",
        "what's in your toolbox", "who are you and what do you do"])
    if rng.random() < 0.3:
        core = rng.choice(["list your features", "give me a rundown of your abilities",
                           "explain what you can do", "show me your commands",
                           "tell me what you support", "describe your abilities"])
        return _polite(rng, core), {}
    return _polite(rng, core, kind="q"), {}


def g_answer(rng):
    q = rng.choice(QUESTIONS) + rng.choice(["", "", "?"])
    return (rng.choice(CALL) + q).strip(), {"question": q}  # the question without "hey r2"


def g_clarify(rng):
    u, q = rng.choice(VAGUE)
    return _polite(rng, u, kind="stmt"), {"question": q}


def g_unsupported(rng):
    reason, utts = rng.choice(list(OUT_OF_SCOPE.items()))
    return _polite(rng, rng.choice(utts)), {"reason": reason}


GENERATORS = {  # intent -> (generator, weight)
    "git_status": (g_git_status, 8), "git_diff_summary": (g_git_diff, 7),
    "search_files": (g_search, 12), "read_file": (g_read, 10), "open_app": (g_open_app, 7),
    "remember_preference": (g_remember, 9), "recall_preference": (g_recall, 7),
    "general_help": (g_help, 5), "answer_question": (g_answer, 9),
    "clarify_request": (g_clarify, 7), "unsupported": (g_unsupported, 14),
}


# ------------------------------------------------------------------ decontamination
def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s./*+-]", " ", text.lower()).split())


def words(text: str) -> set[str]:
    return set(norm(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def reference_utterances() -> list[str]:
    refs = [u for u, _ in FEW_SHOT]
    for name in EVAL_SETS:
        path = ROOT / "data" / name
        if path.exists():
            lines = path.read_text().splitlines()
            refs += [json.loads(line)["utterance"] for line in lines if line]
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700, help="training examples")
    ap.add_argument("--valid", type=int, default=70)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synth")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    system = system_prompt_for(load_settings(), default_registry())
    refs = reference_utterances()
    ref_norm = {norm(r) for r in refs}
    ref_words = [words(r) for r in refs]

    intents = list(GENERATORS)
    weights = [GENERATORS[i][1] for i in intents]
    seen: set[str] = set()
    rows: list[dict] = []
    dropped = Counter()
    target = args.n + args.valid
    attempts = 0
    while len(rows) < target and attempts < target * 200:
        attempts += 1
        intent = rng.choices(intents, weights)[0]
        utt, params = GENERATORS[intent][0](rng)
        key = norm(utt)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if key in ref_norm:
            dropped["equals_eval_or_fewshot"] += 1
            continue
        w = words(utt)
        if any(jaccard(w, r) >= JACCARD_MAX for r in ref_words):
            dropped["near_eval_or_fewshot"] += 1
            continue
        seen.add(key)
        rows.append({"intent": intent, "utterance": utt, "parameters": params})

    rng.shuffle(rows)
    valid, train = rows[: args.valid], rows[args.valid:]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def record(r):
        return {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": r["utterance"]},
            {"role": "assistant", "content": json.dumps(
                {"intent": r["intent"], "parameters": r["parameters"]}, ensure_ascii=False)},
        ]}

    for name, part in (("train", train), ("valid", valid)):
        with (args.out_dir / f"{name}.jsonl").open("w") as fh:
            for r in part:
                fh.write(json.dumps(record(r), ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed, "train": len(train), "valid": len(valid),
        "by_intent": dict(Counter(r["intent"] for r in train)),
        "dropped": dict(dropped), "jaccard_max": JACCARD_MAX, "decontaminated_against":
        ["prompt few-shot"] + EVAL_SETS,
        "examples": [f"{r['utterance']} -> {r['intent']} {r['parameters']}" for r in train[:15]],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "examples"}, indent=2))
    for e in manifest["examples"]:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
