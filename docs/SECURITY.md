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
| Other local software / web pages calling the HTTP API | Bearer token (0600 file, rotated per start); loopback-only bind |
| Browser CSRF / DNS rebinding to `127.0.0.1` | `Host` allowlist, `Origin`/`Sec-Fetch-Site` rejected, JSON-only bodies (forces a preflight that is never answered) |
| Replayed or swapped confirmation | Single-use, 120 s `confirmation_id` bound to a digest of the exact decision; wrong/missing id cancels |
| Symlink escape / check-then-open race | `O_NOFOLLOW` open + re-check of the opened fd's real path; search never follows symlinked dirs |
| Fake output lines via crafted filenames | Control characters in listed names are replaced |
| Multi-step run bypassing checks | Every step goes through the same `ToolExecutor` path; plans validated all-or-nothing; stop on first failure; ≤5 steps + time budget |
| Prompt injection steering later steps | Plans are made once, before any tool runs; tool output is never fed back into planning |
| Asking to confirm something that will be refused | Tool prechecks (e.g. app allowlist) run before the confirmation prompt |

## Fail closed

If JSON parse, permission check, or path validation fails → do not execute.

## Data retention

- Preferences: local SQLite only, with explicit remember confirmation.
- Audio: temp WAV for transcription; not retained by default.
- Conversation: in-process bounded history only unless you add persistence later.
- Multi-step runs (`agent.persist_runs`, default on): `runs.db` next to the preference DB
  keeps the request text, run/step status, step intents and parameters, and each step's
  short spoken summary. Tool output (file contents, diffs, full answers) is never written to
  disk. It exists only in memory and in the API response. Set `agent.persist_runs = false`
  to keep nothing.

## Local HTTP API

`r2 serve` (used by the SwiftUI app) is guarded before any routing, and fails closed:

1. **Loopback only.** Binding to any non-loopback address is refused.
2. **`Host` allowlist.** Only `127.0.0.1:<port>`, `localhost:<port>` and `[::1]:<port>`
   are accepted. A DNS-rebinding page sends its own hostname here, and gets 403.
3. **No browsers.** Any `Origin` header, or `Sec-Fetch-Site: cross-site|same-site`, gets 403.
   The native client sends neither. `OPTIONS` (CORS preflight) gets 405 with no CORS headers.
4. **Bearer token.** Every endpoint except `/health` needs `Authorization: Bearer <token>`.
   The token is 32 random bytes, written atomically with mode 0600 to
   `~/.local/share/narrow-ai/api-token` before the server listens, and deleted on exit
   (SIGINT or SIGTERM). Set `security.api_token_path` to move it.
5. **JSON-only bodies.** POSTs must be `application/json` (`audio/wav` for `/ask_audio`).
   That rules out the HTML-form / `text/plain` "simple requests" used for CSRF.

**Confirmations are bound to the prompt.** When a privileged action is parked, the response
carries a random `confirmation_id` (single use, `security.confirmation_ttl_sec`, default 120 s).
The id is tied to a SHA-256 digest of the exact decision. Over HTTP, an approval only counts
if it carries that id: `/confirm`, or a typed or spoken "yes" sent with the id. A missing,
wrong, expired or tampered id **cancels** the action. `/status` never shows the id. The
in-process terminal UI accepts a plain "yes", because the person typing is the one who saw
the prompt. Every outcome (approved, mismatch, expired, cancelled, superseded) is audited.

**Trust boundary, stated honestly.** Any process running as the same macOS user can read a
0600 file, so it can read the token. That puts same-user processes inside the trust boundary;
they could equally read the preference DB or edit R2's config. The token stops web pages,
other users on the machine, and accidental clients. A Unix domain socket was considered and
deferred: `URLSession` can't talk to one (it would need Network.framework), and it has the
same same-user limit.

