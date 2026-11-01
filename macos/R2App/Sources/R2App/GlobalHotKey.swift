import AppKit
import Carbon.HIToolbox
import ServiceManagement
import SwiftUI

/// System-wide ⌥Space, like holding the Siri key: bring R2 forward and start listening;
/// press again to stop. Uses Carbon's RegisterEventHotKey, which needs no Accessibility
/// permission (unlike a global NSEvent monitor).
final class GlobalHotKey {
    private static var current: GlobalHotKey?
    private var hotKey: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private let action: () -> Void

    private init(action: @escaping () -> Void) { self.action = action }

    /// Installs (or replaces) the app's single hot key. Returns false if macOS refused it,
    /// e.g. because another app already owns ⌥Space.
    @discardableResult
    static func install(keyCode: Int = kVK_Space, modifiers: Int = optionKey,
                        action: @escaping () -> Void) -> Bool {
        uninstall()
        let hk = GlobalHotKey(action: action)
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                 eventKind: UInt32(kEventHotKeyPressed))
        let installed = InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in
            DispatchQueue.main.async { GlobalHotKey.current?.action() }
            return noErr
        }, 1, &spec, nil, &hk.handler)
        guard installed == noErr else { return false }

        let id = EventHotKeyID(signature: OSType(0x5232_4B59), id: 1)   // 'R2KY'
        let registered = RegisterEventHotKey(UInt32(keyCode), UInt32(modifiers), id,
                                             GetApplicationEventTarget(), 0, &hk.hotKey)
        guard registered == noErr else {
            if let h = hk.handler { RemoveEventHandler(h) }
            return false
        }
        current = hk
        return true
    }

    static func uninstall() {
        guard let hk = current else { return }
        if let k = hk.hotKey { UnregisterEventHotKey(k) }
        if let h = hk.handler { RemoveEventHandler(h) }
        current = nil
    }
}

/// Wires the hot key and window-opening into the client once a scene appears.
struct SummonSupport: ViewModifier {
    @Environment(\.openWindow) private var openWindow
    let client: R2Client

    func body(content: Content) -> some View {
        content.onAppear {
            client.openMainWindow = { openWindow(id: "main") }
            client.applyHotKey()
        }
    }
}

/// "Open at login" via SMAppService. Works for the bundled app (build/R2.app or /Applications).
enum LoginItem {
    static var isEnabled: Bool { SMAppService.mainApp.status == .enabled }

    /// Returns an error message, or nil on success.
    static func set(_ on: Bool) -> String? {
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            return nil
        } catch {
            return error.localizedDescription
        }
    }
}
