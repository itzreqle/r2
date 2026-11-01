import SwiftUI

struct ContentView: View {
    @EnvironmentObject var r2: R2Client
    @State private var input = ""

    var body: some View {
        VStack(spacing: 18) {
            OrbView(phase: r2.phase)
                .frame(width: 150, height: 150)
                .padding(.top, 28)
                .onTapGesture { r2.toggleListening() }
                .help("Click to talk (click again to stop)")

            Text(r2.transcript.isEmpty ? "Ask R2 about your workspace" : "“\(r2.transcript)”")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Text(r2.reply)
                .font(.title3.weight(.semibold))
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .padding(.horizontal)

            if !r2.detail.isEmpty {
                ScrollView {
                    Text(r2.detail)
                        .font(.system(.footnote, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 160)
                .padding(10)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal)
            }

            if r2.phase == .confirm {
                HStack {
                    Button("Cancel", role: .cancel) { r2.cancel() }
                    Button("Yes, do it") { r2.confirm() }
                        .keyboardShortcut(.defaultAction)
                        .buttonStyle(.borderedProminent)
                }
            }

            Spacer(minLength: 0)

            HStack(spacing: 8) {
                TextField("Type a request…", text: $input)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { r2.ask(input); input = "" }
                Button {
                    r2.toggleListening()
                } label: {
                    Image(systemName: r2.phase == .listening ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .help("Push to talk")
            }
            .padding(.horizontal)

            HStack {
                Text(r2.metrics.isEmpty ? r2.kvInfo : r2.metrics)
                Spacer()
                Toggle("Speak", isOn: $r2.speakReplies).toggleStyle(.switch).controlSize(.mini)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding([.horizontal, .bottom])
        }
        .background(.ultraThinMaterial)
    }
}

struct CompactView: View {
    @EnvironmentObject var r2: R2Client
    @State private var input = ""

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                OrbView(phase: r2.phase)
                    .frame(width: 56, height: 56)
                    .onTapGesture { r2.toggleListening() }
                Text(r2.reply).font(.callout.weight(.medium)).lineLimit(4)
                Spacer(minLength: 0)
            }
            if r2.phase == .confirm {
                HStack {
                    Button("Cancel") { r2.cancel() }
                    Button("Yes") { r2.confirm() }.buttonStyle(.borderedProminent)
                }
            }
            TextField("Ask R2…", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit { r2.ask(input); input = "" }
            Text(r2.metrics.isEmpty ? r2.kvInfo : r2.metrics)
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(12)
    }
}
