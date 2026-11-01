import Foundation
import Combine

@MainActor
final class AssistantBridge: ObservableObject {
    @Published var stateLabel = "idle"
    @Published var transcript = ""
    @Published var response = ""
    @Published var needsConfirmation = false

    /// Repo root containing pyproject.toml — adjust if you relocate the project.
    var workspacePath: String {
        // Prefer env override for flexibility
        if let env = ProcessInfo.processInfo.environment["NARROW_AI_WORKSPACE"], !env.isEmpty {
            return env
        }
        // Default: parent of macos/
        let thisFile = URL(fileURLWithPath: #filePath)
        // .../narrow-ai-with-kv/macos/NarrowAI/AssistantBridge.swift
        return thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
    }

    private var pendingConfirm = false

    func send(_ text: String) {
        transcript = text
        stateLabel = "thinking"
        needsConfirmation = false
        Task {
            let out = await runOnce(text)
            await handleOutput(out)
        }
    }

    func confirm() {
        guard needsConfirmation else { return }
        stateLabel = "executing"
        Task {
            // Re-send a confirmation via CLI: user types yes in text mode.
            // For the shell we invoke with an env flag if present; otherwise echo yes path:
            let out = await runOnce("yes")
            await handleOutput(out)
        }
    }

    func cancel() {
        needsConfirmation = false
        pendingConfirm = false
        stateLabel = "idle"
        response = "Cancelled."
    }

    private func handleOutput(_ out: String) async {
        response = out
        needsConfirmation = out.localizedCaseInsensitiveContains("Confirm")
        stateLabel = needsConfirmation ? "awaiting_confirmation" : "idle"
    }

    private func runOnce(_ utterance: String) async -> String {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
                let ws = self.workspacePath
                let escaped = utterance.replacingOccurrences(of: "'", with: "'\\''")
                let cmd = "cd '\(ws)' && uv run narrow-ai text --workspace '\(ws)' --once '\(escaped)'"
                proc.arguments = ["-lc", cmd]
                let pipe = Pipe()
                proc.standardOutput = pipe
                proc.standardError = pipe
                do {
                    try proc.run()
                    proc.waitUntilExit()
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    let text = String(data: data, encoding: .utf8) ?? ""
                    cont.resume(returning: text.trimmingCharacters(in: .whitespacesAndNewlines))
                } catch {
                    cont.resume(returning: "Failed to launch Python core: \(error.localizedDescription)")
                }
            }
        }
    }
}
