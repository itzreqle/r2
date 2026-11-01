import AppKit
import SwiftUI

/// `R2 --render-orb <dir>` renders each orb phase off-screen to PNG and exits.
/// Used to check the animation for artefacts and to make README images.
@MainActor
enum OrbSnapshot {
    static func runIfRequested() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--render-orb") else { return }
        let dir = URL(fileURLWithPath: i + 1 < args.count ? args[i + 1] : ".", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let phases: [R2Client.Phase] = [.idle, .listening, .thinking, .speaking, .confirm, .offline]
        for phase in phases {
            for (name, side, compact) in [("", 400.0, false), ("-compact", 96.0, true)] {
                let sim = OrbSimulation()
                var t = Date()
                for f in 0..<300 {   // 5 s of simulated frames at 60 fps
                    t += 1.0 / 60.0
                    let lvl = phase == .listening ? 0.55 + 0.4 * sin(Double(f) / 6) : 0
                    sim.step(to: t, phase: phase, level: lvl, tick: phase == .speaking ? f / 9 : 0)
                }
                let view = Canvas { ctx, size in sim.draw(in: &ctx, size: size, compact: compact) }
                    .frame(width: side, height: side)
                    .background(Color.black)
                let r = ImageRenderer(content: view)
                r.scale = 2
                if let tiff = r.nsImage?.tiffRepresentation,
                   let rep = NSBitmapImageRep(data: tiff),
                   let png = rep.representation(using: .png, properties: [:]) {
                    try? png.write(to: dir.appendingPathComponent("orb-\(phase.rawValue)\(name).png"))
                }
            }
        }
        print("rendered orb phases to \(dir.path)")
        exit(0)
    }
}
