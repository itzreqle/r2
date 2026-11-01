# R2 for macOS

`R2App/` is a SwiftUI app (Swift Package, macOS 14+). It is a thin client: speech
recognition, intent decisions, permissions and the KV cache all live in the Python core,
which the app starts and talks to over `http://127.0.0.1:8765`.

## Build and run

```bash
cd macos/R2App
./build_app.sh --open        # release build → build/R2.app, then launch
./build_app.sh --install     # also copy to ~/Applications/R2.app
swift run                    # unbundled dev run
```

`build_app.sh` builds the binary, makes the icon, writes `Info.plist` (microphone usage text
and the repo path the app uses to start the core), and ad-hoc signs the bundle.

## What's inside

| File | Role |
| --- | --- |
| `R2App.swift` | window, menus and shortcuts, Settings, menu-bar extra |
| `R2Client.swift` | state machine (offline → idle → listening → thinking → speaking / confirm) and HTTP calls |
| `OrbView.swift` | the animated orb (Canvas + TimelineView, eased physics, per-phase palettes) |
| `ContentView.swift` | main window, compact menu-bar panel, Settings form |
| `AudioCapture.swift` | microphone via AVAudioEngine, live level, WAV encoding |
| `Speaker.swift` | Kokoro voice from the core (`POST /tts`), metered for the orb; system voice fallback |
| `GlobalHotKey.swift` | ⌥Space summon shortcut (Carbon, no Accessibility permission), open at login |
| `CoreLauncher.swift` | starts `uv run r2 --mute serve`, logs to `~/Library/Logs/R2/core.log` |
| `OrbSnapshot.swift` | `R2 --render-orb <dir>` writes orb PNGs for the docs |

Launch options: `R2 --ask "find the swift files"` sends one request once the core is ready,
`R2 --settings` opens Settings, and `R2 --demo` cycles through every orb state without the core.
The README screenshots were taken this way, with `screencapture -l <window id>`.

## HTTP API (served by `uv run r2 serve`)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/status`, `/voices` | liveness, model + KV cache info, voice list |
| POST | `/ask` `{"text"}` | text request → reply, speech, intent, metrics |
| POST | `/confirm`, `/cancel`, `/reset` | answer a pending confirmation, clear history |
| POST | `/ask_audio` (raw WAV) | Whisper → R2 |
| POST | `/tts` `{"text","voice","speed"}` | Kokoro speech as WAV |

The server binds to 127.0.0.1 only.
