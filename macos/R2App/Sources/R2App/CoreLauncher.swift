import Foundation

/// Starts / stops the Python core (`uv run r2 serve`) so R2.app works on its own.
final class CoreLauncher {
    private(set) var process: Process?
    let logURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/R2", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("core.log")
    }()

    var isRunning: Bool { process?.isRunning ?? false }

    func start(repo: String) throws {
        guard !isRunning else { return }
        guard FileManager.default.fileExists(atPath: repo + "/pyproject.toml") else {
            throw NSError(domain: "R2", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "R2 project not found at \(repo). Set it in Settings (⌘,)."])
        }
        let quoted = "'" + repo.replacingOccurrences(of: "'", with: "'\\''") + "'"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        // Login shell so `uv` from the user's PATH is found; HF offline = fully local.
        p.arguments = ["-lc", "cd \(quoted) && HF_HUB_OFFLINE=1 exec uv run r2 --mute serve"]
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        p.standardOutput = log
        p.standardError = log
        try p.run()
        process = p
    }

    func stop() {
        guard let p = process, p.isRunning else { return }
        p.terminate()
        process = nil
    }
}
