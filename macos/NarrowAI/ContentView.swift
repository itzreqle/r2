import SwiftUI

struct ContentView: View {
    @EnvironmentObject var bridge: AssistantBridge
    @State private var input = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Narrow AI")
                .font(.headline)
            Text(bridge.stateLabel)
                .font(.caption)
                .foregroundStyle(.secondary)

            TextField("Ask a developer question…", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit { send() }

            HStack {
                Button("Send") { send() }
                    .keyboardShortcut(.return, modifiers: [])
                Button("Confirm") { bridge.confirm() }
                    .disabled(!bridge.needsConfirmation)
                Button("Cancel") { bridge.cancel() }
            }

            if !bridge.transcript.isEmpty {
                GroupBox("You") {
                    Text(bridge.transcript)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            GroupBox("Assistant") {
                ScrollView {
                    Text(bridge.response.isEmpty ? "—" : bridge.response)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 120, maxHeight: 220)
            }

            Divider()
            Text("Workspace: \(bridge.workspacePath)")
                .font(.caption2)
                .lineLimit(2)
        }
        .padding()
    }

    private func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        bridge.send(text)
        input = ""
    }
}
