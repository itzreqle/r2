# macOS SwiftUI shell

Minimal menu-bar client that talks to the Python core via a local subprocess (`uv run narrow-ai text --once`).

## Open in Xcode

1. Create a new macOS App (SwiftUI) named `NarrowAI`, or open the sources under `NarrowAI/`.
2. Set deployment target to macOS 14+.
3. Enable App Sandbox carefully: for MVP development, run without sandbox or allow outbound/network none + subprocess. Prefer **non-sandboxed developer builds** first so `uv`/`python` can run.
4. Grant Microphone if you wire voice later.

## Protocol

The UI runs:

```bash
uv run narrow-ai text --workspace <path> --once "<utterance>"
```

and displays stdout. Confirmation flows use a second invocation after the user taps Confirm.

This keeps the Python core independently testable; the Swift app is a thin shell.
