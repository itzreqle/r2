import AppKit
import Foundation
import SwiftUI

/// App state + bridge to the Python core (`r2 serve` on 127.0.0.1:8765).
/// All intelligence, permissions and the KV cache live in Python; the app records
/// audio, speaks, animates, and forwards input.
@MainActor
final class R2Client: ObservableObject {
    enum Phase: String { case offline, starting, idle, listening, thinking, speaking, confirm, error }

    struct Exchange: Identifiable, Equatable {
        let id = UUID()
        let user: String
        let reply: String
    }

    @Published var phase: Phase = .offline
    @Published var caption = ""                 // what R2 heard / you typed
    @Published var reply = ""                   // R2's short spoken answer
    @Published var detail = ""                  // full tool output
    @Published var intent = ""
    @Published var latency = ""
    @Published var kvInfo = ""
    @Published var modelName = ""
    @Published var level: Float = 0             // live mic level for the orb
    @Published var speechTick = 0               // increments per spoken word
    @Published var history: [Exchange] = []
    @Published var lastError = ""

    @Published var speakReplies: Bool { didSet { defaults.set(speakReplies, forKey: "speakReplies") } }
    @Published var playSounds: Bool { didSet { defaults.set(playSounds, forKey: "playSounds") } }
    @Published var autoStartCore: Bool { didSet { defaults.set(autoStartCore, forKey: "autoStartCore") } }
    @Published var repoPath: String { didSet { defaults.set(repoPath, forKey: "repoPath") } }
    @Published var naturalVoice: Bool { didSet { defaults.set(naturalVoice, forKey: "naturalVoice") } }
    @Published var voiceName: String { didSet { defaults.set(voiceName, forKey: "voiceName") } }
    @Published var speechSpeed: Double { didSet { defaults.set(speechSpeed, forKey: "speechSpeed") } }
    @Published var hotKeyEnabled: Bool {
        didSet { defaults.set(hotKeyEnabled, forKey: "hotKeyEnabled"); applyHotKey() }
    }
    @Published var hotKeyError = ""

    /// Set by a scene (needs SwiftUI's openWindow); lets the hot key reopen a closed window.
    var openMainWindow: (() -> Void)?

    static let voices: [(id: String, label: String)] = [
        ("af_heart", "Heart — warm, natural"), ("af_bella", "Bella — bright, friendly"),
        ("af_nicole", "Nicole — soft, calm"), ("af_sarah", "Sarah — clear, neutral"),
        ("am_michael", "Michael — warm"), ("am_fenrir", "Fenrir — deep"),
        ("bf_emma", "Emma — British"), ("bm_george", "George — British"),
    ]

    let base = URL(string: "http://127.0.0.1:8765")!
    private let defaults = UserDefaults.standard
    private let audio = AudioCapture()
    private let speaker = Speaker()
    let core = CoreLauncher()
    private var awaitingConfirm = false
    private var bootTask: Task<Void, Never>?

    init() {
        defaults.register(defaults: ["speakReplies": true, "playSounds": true, "autoStartCore": true,
                                     "naturalVoice": true, "voiceName": "af_heart", "speechSpeed": 1.0,
                                     "hotKeyEnabled": true])
        hotKeyEnabled = defaults.bool(forKey: "hotKeyEnabled")
        speakReplies = defaults.bool(forKey: "speakReplies")
        playSounds = defaults.bool(forKey: "playSounds")
        autoStartCore = defaults.bool(forKey: "autoStartCore")
        repoPath = defaults.string(forKey: "repoPath") ?? Self.defaultRepoPath()
        naturalVoice = defaults.bool(forKey: "naturalVoice")
        voiceName = defaults.string(forKey: "voiceName") ?? "af_heart"
        speechSpeed = defaults.double(forKey: "speechSpeed")

        audio.onLevel = { [weak self] lvl in
            DispatchQueue.main.async { self?.level = lvl }
        }
        speaker.onStart = { [weak self] in self?.phase = .speaking }
        speaker.onWord = { [weak self] in self?.speechTick &+= 1 }
        speaker.onLevel = { [weak self] lvl in
            guard let self, self.phase == .speaking || lvl == 0 else { return }
            self.level = lvl
        }
        speaker.onFinish = { [weak self] in
            guard let self, self.phase == .speaking else { return }
            self.phase = self.awaitingConfirm ? .confirm : .idle
        }
        if CommandLine.arguments.contains("--demo") { runDemo() } else { boot() }
    }

    // MARK: - Summon (⌥Space)

    func applyHotKey() {
        guard hotKeyEnabled else {
            GlobalHotKey.uninstall()
            hotKeyError = ""
            return
        }
        let ok = GlobalHotKey.install { [weak self] in self?.summon() }
        hotKeyError = ok ? "" : "⌥Space is taken by another app."
    }

    /// Bring R2 forward and toggle listening, like pressing the Siri key.
    func summon() {
        NSApp.activate(ignoringOtherApps: true)
        if let w = NSApp.windows.first(where: { $0.identifier?.rawValue == "main" }) {
            w.makeKeyAndOrderFront(nil)
        } else {
            openMainWindow?()
        }
        switch phase {
        case .idle, .speaking, .error: startListening()
        case .listening: stopListening()
        default: break   // offline/starting/thinking/confirm: just show the window
        }
    }

    /// `R2 --demo`: cycles every orb phase with a simulated level (visual QA, no core).
    private func runDemo() {
        let phases: [Phase] = [.idle, .listening, .thinking, .speaking, .confirm, .offline]
        Task {
            var i = 0
            while true {
                let p = phases[i % phases.count]
                phase = p
                reply = "Demo: \(p.rawValue)"
                for f in 0..<150 {   // ~2.5 s per phase
                    level = (p == .listening || p == .speaking) ? Float(0.5 + 0.45 * sin(Double(f) / 5)) : 0
                    if p == .speaking && f % 9 == 0 { speechTick &+= 1 }
                    try? await Task.sleep(nanoseconds: 16_000_000)
                }
                i += 1
            }
        }
    }

    /// Repo path baked into Info.plist by build_app.sh, else derived from the binary location.
    static func defaultRepoPath() -> String {
        if let p = Bundle.main.object(forInfoDictionaryKey: "R2RepoPath") as? String, !p.isEmpty { return p }
        var url = Bundle.main.bundleURL
        for _ in 0..<6 {
            url.deleteLastPathComponent()
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("pyproject.toml").path) {
                return url.path
            }
        }
        return FileManager.default.homeDirectoryForCurrentUser.path
    }

    // MARK: - Core lifecycle

    func boot() {
        bootTask?.cancel()
        bootTask = Task {
            if await refreshStatus() { return greet() }
            guard autoStartCore else {
                phase = .offline
                reply = "R2 core is offline"
                detail = "Start it with:  uv run r2 serve   — or enable auto-start in Settings."
                return
            }
            phase = .starting
            reply = "Waking up…"
            caption = "Loading the on-device model"
            do { try core.start(repo: repoPath) } catch {
                phase = .error; reply = "Couldn't start R2"; detail = error.localizedDescription; return
            }
            for _ in 0..<180 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if Task.isCancelled { return }
                if await refreshStatus() { return greet() }
                if !core.isRunning { break }
            }
            phase = .error
            reply = "R2 core didn't start"
            detail = "See the log: \(core.logURL.path)"
        }
    }

    func restartCore() {
        core.stop()
        phase = .starting
        Task {
            try? await Task.sleep(nanoseconds: 800_000_000)
            boot()
        }
    }

    func shutdown() {
        speaker.stop()
        _ = audio.stop()
        core.stop()
    }

    private func greet() {
        phase = .idle
        caption = ""
        reply = "Hi, I'm R2."
        detail = ""
        if playSounds { NSSound(named: "Glass")?.play() }
        // `R2 --ask "find the swift files"`: send one request at launch (scripting, screenshots).
        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--ask"), i + 1 < args.count {
            let text = args[i + 1]
            Task {
                try? await Task.sleep(nanoseconds: 600_000_000)
                ask(text)
            }
        }
    }

    @discardableResult
    func refreshStatus() async -> Bool {
        do {
            let s = try await call("status", get: true, timeout: 2)
            if let kv = s["kv_cache"] as? [String: Any] {
                let mode = kv["mode"] as? String ?? "?"
                let tok = kv["prefix_tokens"] as? Int ?? 0
                kvInfo = mode == "none" ? "KV cache off" : "KV cache · \(tok) tokens"
            }
            if let m = s["model"] as? String {
                modelName = m.split(separator: "/").last.map(String.init)?
                    .replacingOccurrences(of: "-4bit", with: "") ?? m
            }
            return true
        } catch {
            return false
        }
    }

    // MARK: - Requests

    private func call(_ path: String, body: [String: Any] = [:], get: Bool = false,
                      timeout: TimeInterval = 120) async throws -> [String: Any] {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.timeoutInterval = timeout
        if !get {
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
    }

    func ask(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, phase != .offline, phase != .starting else { return }
        speaker.stop()
        caption = t
        send("ask", body: ["text": t, "speak": false], user: t)
    }

    func confirm() { send("confirm", body: ["speak": false], user: "Yes") }

    func cancel() {
        if phase == .listening { _ = audio.stop(); level = 0 }
        send("cancel", body: [:], user: "Cancel")
    }

    func reset() {
        speaker.stop()
        history.removeAll()
        caption = ""; reply = "Hi, I'm R2."; detail = ""; intent = ""; latency = ""
        awaitingConfirm = false
        if phase != .offline && phase != .starting { phase = .idle }
        Task { _ = try? await call("reset") }
    }

    private func send(_ path: String, body: [String: Any], user: String) {
        phase = .thinking
        Task {
            do {
                apply(try await call(path, body: body), user: user)
            } catch {
                phase = .offline
                reply = "Lost connection to R2 core"
                detail = "Restart it from the R2 menu (⌘⇧R)."
            }
        }
    }

    // MARK: - Voice

    func toggleListening() {
        phase == .listening ? stopListening() : startListening()
    }

    func startListening() {
        guard phase != .offline, phase != .starting, phase != .listening else { return }
        speaker.stop()
        AudioCapture.requestPermission { [weak self] ok in
            guard let self else { return }
            guard ok else {
                self.phase = .error
                self.reply = "Microphone access is off"
                self.detail = "System Settings → Privacy & Security → Microphone → enable R2."
                return
            }
            do {
                try self.audio.start()
                if self.playSounds { NSSound(named: "Tink")?.play() }
                self.phase = .listening
                self.caption = "Listening…"
                self.reply = ""
                self.detail = ""
            } catch {
                self.phase = .error
                self.reply = "Couldn't start the microphone"
                self.detail = error.localizedDescription
            }
        }
    }

    func stopListening() {
        guard phase == .listening else { return }
        let wav = audio.stop()
        level = 0
        if playSounds { NSSound(named: "Pop")?.play() }
        guard let wav else {
            phase = .idle
            caption = ""
            reply = "I didn't catch that."
            return
        }
        phase = .thinking
        caption = "…"
        Task {
            do {
                var req = URLRequest(url: base.appendingPathComponent("ask_audio"))
                req.url = URL(string: base.absoluteString + "/ask_audio?speak=0")
                req.httpMethod = "POST"
                req.setValue("audio/wav", forHTTPHeaderField: "Content-Type")
                req.timeoutInterval = 120
                let (data, _) = try await URLSession.shared.upload(for: req, from: wav)
                let r = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
                let heard = r["transcript"] as? String ?? ""
                caption = heard.isEmpty ? "" : heard
                apply(r, user: heard)
            } catch {
                phase = .offline
                reply = "Lost connection to R2 core"
            }
        }
    }

    // MARK: - Voice output

    func say(_ text: String) {
        speaker.speak(text, natural: naturalVoice, voice: voiceName, speed: speechSpeed, server: base)
    }

    func previewVoice() {
        guard phase == .idle || phase == .speaking else { return }
        phase = .speaking
        say("Hi, I'm R2. This is how I sound.")
    }

    // MARK: - Responses

    private func apply(_ r: [String: Any], user: String) {
        if let err = r["error"] as? String {
            phase = .error; reply = "Something went wrong"; detail = err
            return
        }
        let speech = (r["speech"] as? String) ?? (r["text"] as? String) ?? ""
        let text = (r["text"] as? String) ?? ""
        let needsConfirm = (r["needs_confirmation"] as? Bool) ?? false
        withAnimation(.smooth(duration: 0.35)) {
            reply = speech
            // A confirmation's text only restates the question; the buttons say the rest.
            detail = (text == speech || needsConfirm) ? "" : text
        }
        intent = (r["intent"] as? String ?? "").replacingOccurrences(of: "_", with: " ")
        if let m = r["metrics"] as? [String: Any], let think = m["think_ms"] as? Double {
            let ttft = m["ttft_ms"] as? Double ?? 0
            latency = "\(Int(ttft)) ms first token · \(Int(think)) ms total"
        } else {
            latency = ""
        }
        if !user.isEmpty && !speech.isEmpty {
            history.append(Exchange(user: user, reply: speech))
            if history.count > 20 { history.removeFirst() }
        }
        awaitingConfirm = needsConfirm
        if speakReplies && !speech.isEmpty {
            phase = .speaking
            say(speech)
        } else {
            phase = awaitingConfirm ? .confirm : .idle
        }
    }
}
