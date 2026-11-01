import Foundation
import SwiftUI

/// Thin client for the Python core (`uv run r2 serve`). All intelligence, permissions
/// and the KV cache live in Python; this app only renders state and forwards input.
@MainActor
final class R2Client: ObservableObject {
    enum Phase: String { case offline, idle, listening, thinking, confirm, error }

    @Published var phase: Phase = .offline
    @Published var transcript = ""
    @Published var reply = "Start the core with:  uv run r2 serve"
    @Published var detail = ""
    @Published var metrics = ""
    @Published var kvInfo = ""
    @Published var speakReplies = true

    let base = URL(string: "http://127.0.0.1:8765")!
    private var poller: Timer?

    init() {
        refreshStatus()
        poller = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            Task { @MainActor in
                if self?.phase == .offline { self?.refreshStatus() }
            }
        }
    }

    var menuIcon: String {
        switch phase {
        case .listening: return "waveform.circle.fill"
        case .thinking: return "ellipsis.circle"
        case .confirm: return "questionmark.circle"
        case .offline, .error: return "exclamationmark.circle"
        case .idle: return "circle.hexagongrid.circle"
        }
    }

    // MARK: - API
    private func call(_ path: String, body: [String: Any] = [:], get: Bool = false) async throws -> [String: Any] {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.timeoutInterval = 120
        if !get {
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, _) = try await URLSession.shared.data(for: req)
        return (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
    }

    func refreshStatus() {
        Task {
            do {
                let s = try await call("status", get: true)
                if let kv = s["kv_cache"] as? [String: Any] {
                    let mode = kv["mode"] as? String ?? "?"
                    let tok = kv["prefix_tokens"] as? Int ?? 0
                    let src = kv["source"] as? String ?? "?"
                    kvInfo = "KV cache \(mode) · \(tok) tokens (\(src))"
                }
                if phase == .offline {
                    phase = .idle
                    reply = "Hi, I'm R2. Ask me about your repo."
                }
            } catch {
                phase = .offline
                reply = "R2 core is offline. Run:  uv run r2 serve"
            }
        }
    }

    func ask(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        transcript = t
        run("ask", body: ["text": t])
    }

    func confirm() { run("confirm") }
    func cancel() { run("cancel") }

    func toggleListening() {
        if phase == .listening {
            phase = .thinking
            run("listen/stop")
        } else {
            Task {
                do {
                    _ = try await call("listen/start")
                    phase = .listening
                    transcript = "Listening…"
                } catch {
                    phase = .error
                    reply = "Could not start microphone: \(error.localizedDescription)"
                }
            }
        }
    }

    private func run(_ path: String, body: [String: Any] = [:]) {
        var b = body
        b["speak"] = speakReplies
        phase = .thinking
        Task {
            do {
                let r = try await call(path, body: b)
                apply(r)
            } catch {
                phase = .offline
                reply = "R2 core is offline. Run:  uv run r2 serve"
            }
        }
    }

    private func apply(_ r: [String: Any]) {
        if let err = r["error"] as? String {
            phase = .error
            reply = err
            return
        }
        if let heard = r["transcript"] as? String, !heard.isEmpty { transcript = heard }
        reply = (r["speech"] as? String) ?? (r["text"] as? String) ?? ""
        let text = (r["text"] as? String) ?? ""
        detail = text == reply ? "" : text
        if let m = r["metrics"] as? [String: Any], let ttft = m["ttft_ms"] as? Double {
            let cached = m["cached_tokens"] as? Int ?? 0
            let intent = r["intent"] as? String ?? "-"
            metrics = "\(intent) · ttft \(Int(ttft)) ms · \(cached) cached tokens"
        } else {
            metrics = ""
        }
        phase = (r["needs_confirmation"] as? Bool ?? false) ? .confirm : .idle
    }
}
