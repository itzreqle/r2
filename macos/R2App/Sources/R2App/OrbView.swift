import SwiftUI

/// Siri-style animated orb. Colour and motion follow R2's phase.
struct OrbView: View {
    var phase: R2Client.Phase
    @State private var spin = false
    @State private var pulse = false

    private var colors: [Color] {
        switch phase {
        case .listening: return [.pink, .red, .orange, .pink]
        case .thinking: return [.purple, .blue, .cyan, .purple]
        case .confirm: return [.yellow, .orange, .yellow]
        case .offline, .error: return [.gray, .secondary, .gray]
        case .idle: return [.cyan, .blue, .teal, .cyan]
        }
    }

    private var speed: Double {
        switch phase {
        case .listening: return 1.2
        case .thinking: return 0.8
        default: return 6
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(AngularGradient(colors: colors, center: .center))
                .blur(radius: 18)
                .opacity(0.8)
                .scaleEffect(pulse ? 1.12 : 0.92)
            Circle()
                .fill(AngularGradient(colors: colors.reversed(), center: .center))
                .rotationEffect(.degrees(spin ? 360 : 0))
                .mask(Circle().padding(10))
                .overlay(Circle().stroke(.white.opacity(0.35), lineWidth: 1).padding(10))
            Text("R2")
                .font(.system(size: 26, weight: .heavy, design: .rounded))
                .foregroundStyle(.white.opacity(0.9))
        }
        .onAppear { restart() }
        .onChange(of: phase) { _, _ in restart() }
    }

    private func restart() {
        spin = false
        pulse = false
        withAnimation(.linear(duration: speed).repeatForever(autoreverses: false)) { spin = true }
        withAnimation(.easeInOut(duration: speed / 2).repeatForever(autoreverses: true)) { pulse = true }
    }
}
