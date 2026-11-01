import SwiftUI

@main
struct NarrowAIApp: App {
    @StateObject private var bridge = AssistantBridge()

    var body: some Scene {
        MenuBarExtra("Narrow AI", systemImage: "waveform.circle") {
            ContentView()
                .environmentObject(bridge)
                .frame(width: 360)
        }
        .menuBarExtraStyle(.window)
    }
}
