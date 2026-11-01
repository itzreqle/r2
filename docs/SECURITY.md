# Security and privacy

## Threat model (MVP)

| Threat | Mitigation |
| ------ | ---------- |
| Model invents shell commands | No raw shell tool; named allowlisted tools only |
| Path traversal | `path_guard.resolve_in_workspace` |
| Prompt injection via file/test output | Tool results treated as data; permissions in app code |
| Accidental secrets in logs | Redaction filter; refuse `.env` / key files |
| Unintended writes | Destructive intents disabled; confirmation for preferences |
| Network exfiltration | No mandatory cloud APIs at runtime |
| Always-on mic | Push-to-talk only; no recording on launch |

## Fail closed

If JSON parse, permission check, or path validation fails → do not execute.

## Data retention

- Preferences: local SQLite only, with explicit remember confirmation.
- Audio: temp WAV for transcription; not retained by default.
- Conversation: in-process bounded history only unless you add persistence later.
