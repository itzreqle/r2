import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var onQuit: (() -> Void)?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // When run unbundled (swift run) make it a regular foreground app.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        NSApp.appearance = NSAppearance(named: .darkAqua)
    }

    func applicationWillTerminate(_ notification: Notification) {
        onQuit?()
    }
}

@main
struct R2App: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var client = R2Client()

    init() {
        OrbSnapshot.runIfRequested()
    }

    var body: some Scene {
        Window("R2", id: "main") {
            ContentView()
                .environmentObject(client)
                .frame(minWidth: 440, idealWidth: 480, minHeight: 640, idealHeight: 720)
                .onAppear { appDelegate.onQuit = { [client] in client.shutdown() } }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: 480, height: 720)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Conversation") { client.reset() }
                    .keyboardShortcut("n")
            }
            CommandMenu("Assistant") {
                Button(client.phase == .listening ? "Stop Listening" : "Start Listening") {
                    client.toggleListening()
                }
                .keyboardShortcut("l")
                Button("Confirm Action") { client.confirm() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(client.phase != .confirm)
                Button("Cancel") { client.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                Divider()
                Toggle("Speak Replies", isOn: $client.speakReplies)
                    .keyboardShortcut("s", modifiers: [.command, .shift])
                Toggle("Play Sounds", isOn: $client.playSounds)
                Divider()
                Button("Restart R2 Core") { client.restartCore() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                Button("Show Core Log") { NSWorkspace.shared.open(client.core.logURL) }
            }
        }

        Settings {
            SettingsView()
                .environmentObject(client)
        }

        MenuBarExtra {
            CompactView()
                .environmentObject(client)
        } label: {
            Image(systemName: client.menuIcon)
        }
        .menuBarExtraStyle(.window)
    }
}

extension R2Client {
    var menuIcon: String {
        switch phase {
        case .listening: return "waveform"
        case .thinking, .starting: return "ellipsis.circle"
        case .speaking: return "waveform.circle"
        case .confirm: return "questionmark.circle"
        case .offline, .error: return "exclamationmark.circle"
        case .idle: return "circle.circle"
        }
    }
}
