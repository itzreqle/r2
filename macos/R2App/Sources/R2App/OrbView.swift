import SwiftUI

/// R2's live orb.
///
/// Rendered every display frame (TimelineView + Canvas). A small physics state
/// (`OrbSimulation`) eases between per-phase targets so every transition is smooth:
///
/// * idle       — slow breathing, cool palette
/// * listening  — reacts to the live microphone level (waveforms + swelling blobs)
/// * thinking   — fast swirl with a rotating light ring
/// * speaking   — pulses on every spoken word
/// * confirm    — warm amber, waiting for you
/// * offline    — dim and almost still
struct OrbView: View {
    var phase: R2Client.Phase
    /// Live input level 0…1 (microphone while listening).
    var level: Float
    /// Increments on every spoken word; each change kicks the orb.
    var speechTick: Int
    var compact = false

    @State private var sim = OrbSimulation()

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas(rendersAsynchronously: false) { ctx, size in
                sim.step(to: timeline.date, phase: phase, level: Double(level), tick: speechTick)
                sim.draw(in: &ctx, size: size, compact: compact)
            }
        }
        .accessibilityLabel("R2 orb, \(phase.rawValue)")
    }
}

// MARK: - Simulation

final class OrbSimulation {
    private var lastDate: Date?
    private var time: Double = 0          // advances with spin speed
    private var clock: Double = 0         // wall clock
    private var energy: Double = 0.1      // overall liveliness 0…~1.6
    private var wave: Double = 0          // waveform amplitude
    private var ring: Double = 0          // thinking ring intensity
    private var spin: Double = 0.3
    private var scale: Double = 0.95
    private var glow: Double = 0.5
    private var mic: Double = 0
    private var kick: Double = 0          // speech impulses
    private var lastTick = 0
    private var palette: [SIMD3<Double>] = Palette.idle

    func step(to date: Date, phase: R2Client.Phase, level: Double, tick: Int) {
        let dt = min(max(date.timeIntervalSince(lastDate ?? date), 0), 1.0 / 20.0)
        lastDate = date
        clock += dt

        // Microphone: fast attack, slow release (like a VU meter).
        let rate = level > mic ? 28.0 : 6.0
        mic += (level - mic) * (1 - exp(-dt * rate))

        // Speech word impulses.
        if tick != lastTick {
            lastTick = tick
            kick = min(kick + 0.9, 1.4)
        }
        kick *= exp(-dt * 5.5)

        let breathe = 0.5 + 0.5 * sin(clock * 1.1)
        var tEnergy = 0.12, tWave = 0.0, tRing = 0.0, tSpin = 0.25, tScale = 0.94, tGlow = 0.45
        var tPalette = Palette.idle
        switch phase {
        case .idle:
            tEnergy = 0.30 + 0.10 * breathe; tScale = 0.94 + 0.03 * breathe; tGlow = 0.75 + 0.20 * breathe
            tSpin = 0.45
        case .listening:
            tEnergy = 0.35 + 1.25 * mic; tWave = 0.15 + 1.3 * mic; tSpin = 0.9 + 1.5 * mic
            tScale = 0.98 + 0.09 * mic; tGlow = 0.65 + 0.5 * mic; tPalette = Palette.listening
        case .thinking:
            tEnergy = 0.55 + 0.12 * sin(clock * 7); tRing = 1; tSpin = 2.6; tScale = 0.95; tGlow = 0.7
            tPalette = Palette.thinking
        case .speaking:
            tEnergy = 0.35 + 0.9 * kick; tWave = 0.1 + 0.9 * kick; tSpin = 1.0 + kick; tScale = 0.97 + 0.07 * kick
            tGlow = 0.6 + 0.4 * kick; tPalette = Palette.speaking
        case .confirm:
            tEnergy = 0.4 + 0.1 * breathe; tSpin = 0.6; tScale = 0.96 + 0.02 * breathe; tGlow = 0.85
            tPalette = Palette.confirm
        case .starting:
            tEnergy = 0.3; tRing = 0.6; tSpin = 1.4; tScale = 0.9; tGlow = 0.45; tPalette = Palette.thinking
        case .offline, .error:
            tEnergy = 0.10; tSpin = 0.10; tScale = 0.9; tGlow = 0.30; tPalette = Palette.offline
        }

        func ease(_ v: inout Double, _ target: Double, _ k: Double) { v += (target - v) * (1 - exp(-dt * k)) }
        ease(&energy, tEnergy, 9)
        ease(&wave, tWave, 12)
        ease(&ring, tRing, 4)
        ease(&spin, tSpin, 3)
        ease(&scale, tScale, 7)
        ease(&glow, tGlow, 4)
        for i in palette.indices {
            let target = tPalette[i % tPalette.count]
            palette[i] += (target - palette[i]) * (1 - exp(-dt * 3.5))
        }
        time += dt * spin
    }

    func draw(in ctx: inout GraphicsContext, size: CGSize, compact: Bool) {
        let c = CGPoint(x: size.width / 2, y: size.height / 2)
        let R = min(size.width, size.height) * (compact ? 0.40 : 0.33) * scale
        let sphere = Path(ellipseIn: CGRect(x: c.x - R, y: c.y - R, width: 2 * R, height: 2 * R))
        let col = palette.map { Color(red: $0.x, green: $0.y, blue: $0.z) }

        // 1. Halo bleeding into the black background.
        ctx.drawLayer { layer in
            layer.blendMode = .plusLighter
            for (i, k) in [(0, 1.9), (2, 1.45)] {
                let r = R * k * (1 + 0.15 * energy)
                layer.fill(
                    Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                    with: .radialGradient(
                        Gradient(colors: [col[i].opacity(0.45 * glow), col[i].opacity(0.12 * glow), .clear]),
                        center: c, startRadius: R * 0.6, endRadius: r))
            }
        }

        // 2. Dark glass body.
        ctx.fill(sphere, with: .radialGradient(
            Gradient(colors: [Color(white: 0.10), Color(white: 0.03), .black]),
            center: CGPoint(x: c.x - R * 0.25, y: c.y - R * 0.35), startRadius: 0, endRadius: R * 1.25))

        // 3. Living colour blobs (additive, blurred, clipped to the sphere).
        ctx.drawLayer { layer in
            layer.clip(to: sphere)
            layer.addFilter(.blur(radius: R * 0.20))
            layer.blendMode = .plusLighter
            let blobs: [(Double, Double, Double, Double, Int)] = [
                // speed, orbit freq, size freq, phase, colour
                (0.9, 0.7, 1.3, 0.0, 0), (-1.2, 0.9, 1.1, 2.1, 1), (1.5, 1.3, 0.8, 4.2, 2),
                (-0.7, 0.5, 1.7, 1.0, 3), (1.1, 1.7, 1.2, 5.3, 1),
            ]
            for (speed, of, sf, ph, ci) in blobs {
                let a = time * speed + ph
                let orbit = R * (0.22 + 0.16 * sin(time * of + ph)) * (1 + 0.55 * energy)
                let p = CGPoint(x: c.x + cos(a) * orbit, y: c.y + sin(a * 1.13) * orbit * 0.9)
                let r = R * (0.46 + 0.14 * sin(time * sf + ph * 1.7)) * (0.85 + 0.55 * energy)
                layer.fill(
                    Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                    with: .radialGradient(
                        Gradient(colors: [col[ci], col[ci].opacity(0.55), .clear]),
                        center: p, startRadius: 0, endRadius: r))
            }
        }

        // 4. Siri-style waveforms while listening / speaking.
        if wave > 0.01 {
            ctx.drawLayer { layer in
                layer.clip(to: sphere)
                layer.blendMode = .plusLighter
                layer.addFilter(.blur(radius: 1.2))
                for (i, (freq, speed, ph)) in [(1.6, 5.0, 0.0), (2.3, -6.5, 1.3), (1.1, 4.0, 2.6)].enumerated() {
                    var path = Path()
                    let steps = 90
                    for s in 0...steps {
                        let u = Double(s) / Double(steps) * 2 - 1          // -1…1 across the sphere
                        let env = pow(1 - u * u, 1.6)                       // pinned at the edges
                        let y = sin(u * .pi * freq + clock * speed + ph) * R * 0.42 * wave * env
                        let pt = CGPoint(x: c.x + u * R * 0.92, y: c.y + y)
                        s == 0 ? path.move(to: pt) : path.addLine(to: pt)
                    }
                    layer.stroke(path, with: .color(col[(i + 1) % col.count].opacity(0.85)),
                                 style: StrokeStyle(lineWidth: max(1.2, R * 0.025), lineCap: .round))
                }
            }
        }

        // 5. Thinking ring — a rotating band of light on the rim.
        if ring > 0.01 {
            ctx.drawLayer { layer in
                layer.blendMode = .plusLighter
                layer.addFilter(.blur(radius: R * 0.05))
                layer.translateBy(x: c.x, y: c.y)
                layer.rotate(by: .radians(clock * 3.2))
                let rr = R * 0.97
                layer.stroke(
                    Path(ellipseIn: CGRect(x: -rr, y: -rr, width: 2 * rr, height: 2 * rr)),
                    with: .conicGradient(
                        Gradient(colors: [.clear, col[0].opacity(ring), col[2].opacity(ring), .clear, .clear]),
                        center: .zero),
                    lineWidth: R * 0.07)
            }
        }

        // 6. Glass: specular highlight + rim light.
        ctx.drawLayer { layer in
            layer.clip(to: sphere)
            let hl = CGRect(x: c.x - R * 0.62, y: c.y - R * 0.86, width: R * 1.0, height: R * 0.62)
            layer.fill(Path(ellipseIn: hl), with: .radialGradient(
                Gradient(colors: [.white.opacity(0.20), .white.opacity(0.04), .clear]),
                center: CGPoint(x: hl.midX, y: hl.midY), startRadius: 0, endRadius: R * 0.55))
        }
        ctx.stroke(sphere, with: .linearGradient(
            Gradient(colors: [.white.opacity(0.22), .white.opacity(0.02), .white.opacity(0.10)]),
            startPoint: CGPoint(x: c.x, y: c.y - R), endPoint: CGPoint(x: c.x, y: c.y + R)),
            lineWidth: 1)
    }
}

// MARK: - Palettes (RGB 0…1)

enum Palette {
    static let idle: [SIMD3<Double>] = [[0.10, 0.75, 1.00], [0.30, 0.35, 1.00], [0.65, 0.30, 1.00], [0.05, 0.90, 0.85]]
    static let listening: [SIMD3<Double>] = [[1.00, 0.18, 0.58], [0.15, 0.80, 1.00], [0.62, 0.25, 1.00], [1.00, 0.45, 0.25]]
    static let thinking: [SIMD3<Double>] = [[0.62, 0.25, 1.00], [0.20, 0.45, 1.00], [1.00, 0.20, 0.70], [0.10, 0.85, 1.00]]
    static let speaking: [SIMD3<Double>] = [[0.10, 0.85, 1.00], [1.00, 0.25, 0.60], [0.40, 0.40, 1.00], [0.20, 1.00, 0.70]]
    static let confirm: [SIMD3<Double>] = [[1.00, 0.70, 0.10], [1.00, 0.45, 0.10], [1.00, 0.85, 0.40], [0.95, 0.35, 0.25]]
    static let offline: [SIMD3<Double>] = [[0.35, 0.35, 0.40], [0.25, 0.25, 0.30], [0.40, 0.40, 0.45], [0.30, 0.30, 0.35]]
}
