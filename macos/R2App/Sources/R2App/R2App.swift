import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Unbundled SwiftPM executables start as background apps; make R2 a normal app.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct R2App: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var client = R2Client()

    var body: some Scene {
        WindowGroup("R2") {
            ContentView()
                .environmentObject(client)
                .frame(minWidth: 420, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)

        MenuBarExtra {
            CompactView()
                .environmentObject(client)
                .frame(width: 340)
        } label: {
            Image(systemName: client.menuIcon)
        }
        .menuBarExtraStyle(.window)
    }
}
