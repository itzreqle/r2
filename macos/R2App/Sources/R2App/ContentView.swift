import AppKit
import SwiftUI

// MARK: - Main window

struct ContentView: View {
    @EnvironmentObject var r2: R2Client
    @State private var input = ""
    @State private var showDetail = true
    @FocusState private var inputFocused: Bool

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer(minLength: 30)

                OrbView(phase: r2.phase, level: r2.level, speechTick: r2.speechTick)
                    .frame(width: 400, height: 400)          // room for the halo to fade out
                    .padding(.vertical, -60)
                    .contentShape(Circle().inset(by: 110))
                    .onTapGesture { r2.toggleListening() }
                    .help(r2.phase == .listening ? "Click to stop" : "Click to talk (⌘L)")

                captionView
                    .padding(.top, 4)

                replyView
                    .padding(.top, 10)

                if r2.phase == .confirm { confirmBar.padding(.top, 18) }

                if !r2.detail.isEmpty { detailCard.padding(.top, 16) }

                if r2.detail.isEmpty && (r2.phase == .idle || r2.phase == .offline) {
                    SuggestionChips()
                        .padding(.top, 22)
                        .transition(.opacity)
                }

                Spacer(minLength: 16)

                inputBar
                footer
            }
            .padding(.bottom, 14)
        }
        .preferredColorScheme(.dark)
        .animation(.smooth(duration: 0.35), value: r2.phase)
        .animation(.smooth(duration: 0.35), value: r2.detail)
        .onAppear { inputFocused = true }
    }

    // What R2 heard / you typed — like Siri's live caption.
    private var captionView: some View {
        Text(r2.caption.isEmpty ? " " : r2.caption)
            .font(.system(size: 15, weight: .regular, design: .rounded))
            .foregroundStyle(.white.opacity(0.5))
            .lineLimit(2)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 40)
            .contentTransition(.opacity)
            .id("cap-\(r2.caption)")
            .transition(.opacity)
    }

    private var replyView: some View {
        Text(r2.reply)
            .font(.system(size: 26, weight: .semibold, design: .rounded))
            .foregroundStyle(
                LinearGradient(colors: [.white, .white.opacity(0.82)], startPoint: .top, endPoint: .bottom))
            .multilineTextAlignment(.center)
            .lineLimit(4)
            .minimumScaleFactor(0.7)
            .padding(.horizontal, 36)
            .textSelection(.enabled)
            .id("reply-\(r2.reply)")
            .transition(.asymmetric(
                insertion: .opacity.combined(with: .offset(y: 8)).combined(with: .scale(scale: 0.98)),
                removal: .opacity))
    }

    private var confirmBar: some View {
        HStack(spacing: 12) {
            Button("Cancel", role: .cancel) { r2.cancel() }
                .keyboardShortcut(.cancelAction)
            Button("Confirm") { r2.confirm() }
                .keyboardShortcut(.defaultAction)
                .tint(.orange)
                .buttonStyle(.borderedProminent)
        }
        .controlSize(.large)
        .transition(.opacity.combined(with: .scale(scale: 0.95)))
    }

    private var detailCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(r2.intent.capitalized, systemImage: "terminal")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(r2.detail, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                .buttonStyle(.borderless)
                .help("Copy")
                Button { withAnimation { showDetail.toggle() } } label: {
                    Image(systemName: showDetail ? "chevron.up" : "chevron.down")
                }
                .buttonStyle(.borderless)
            }
            if showDetail {
                ScrollView {
                    Text(r2.detail)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.85))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 170)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.06))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08)))
        )
        .padding(.horizontal, 24)
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    // Spotlight-style input capsule.
    private var inputBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "sparkle")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .symbolEffect(.pulse, options: .repeating, isActive: r2.phase == .thinking)

            TextField("Ask R2", text: $input)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .focused($inputFocused)
                .onSubmit {
                    r2.ask(input)
                    input = ""
                }
                .disabled(r2.phase == .offline || r2.phase == .starting)

            Button { r2.toggleListening() } label: {
                Image(systemName: r2.phase == .listening ? "waveform" : "mic.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(r2.phase == .listening ? Color.pink : Color.white.opacity(0.75))
                    .symbolEffect(.variableColor.iterative, options: .repeating,
                                  isActive: r2.phase == .listening)
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Talk (⌘L)")
        }
        .padding(.horizontal, 16)
        .frame(height: 46)
        .background(
            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.07))
                .overlay(Capsule(style: .continuous).strokeBorder(
                    LinearGradient(colors: [.white.opacity(inputFocused ? 0.28 : 0.12), .white.opacity(0.05)],
                                   startPoint: .top, endPoint: .bottom)))
        )
        .padding(.horizontal, 22)
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 6, height: 6)
                .shadow(color: statusColor, radius: 3)
            Text(statusText)
            Spacer()
            Text(r2.latency.isEmpty ? r2.kvInfo : r2.latency)
                .monospacedDigit()
                .contentTransition(.numericText())
        }
        .font(.system(size: 11))
        .foregroundStyle(.white.opacity(0.4))
        .padding(.horizontal, 30)
        .padding(.top, 10)
    }

    private var statusText: String {
        switch r2.phase {
        case .offline: return "Offline"
        case .starting: return "Starting on-device model…"
        case .error: return "Error"
        default: return "On-device · \(r2.modelName.isEmpty ? "local model" : r2.modelName)"
        }
    }

    private var statusColor: Color {
        switch r2.phase {
        case .offline, .error: return .red
        case .starting: return .orange
        default: return .green
        }
    }
}

// MARK: - Menu bar panel

struct CompactView: View {
    @EnvironmentObject var r2: R2Client
    @Environment(\.openWindow) private var openWindow
    @State private var input = ""

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 14) {
                OrbView(phase: r2.phase, level: r2.level, speechTick: r2.speechTick, compact: true)
                    .frame(width: 96, height: 96)
                    .padding(-14)
                    .onTapGesture { r2.toggleListening() }
                VStack(alignment: .leading, spacing: 3) {
                    Text(r2.reply.isEmpty ? "Ask R2" : r2.reply)
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .lineLimit(3)
                    if !r2.caption.isEmpty {
                        Text(r2.caption).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
            }
            if r2.phase == .confirm {
                HStack {
                    Button("Cancel") { r2.cancel() }
                    Button("Confirm") { r2.confirm() }.buttonStyle(.borderedProminent).tint(.orange)
                }
                .controlSize(.small)
            }
            HStack(spacing: 8) {
                TextField("Ask R2", text: $input)
                    .textFieldStyle(.plain)
                    .onSubmit { r2.ask(input); input = "" }
                Button { r2.toggleListening() } label: {
                    Image(systemName: r2.phase == .listening ? "waveform" : "mic.fill")
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background(Capsule().fill(Color.white.opacity(0.08)))
            HStack {
                Button("Open R2") {
                    openWindow(id: "main")
                    NSApp.activate(ignoringOtherApps: true)
                }
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(14)
        .frame(width: 330)
        .background(Color.black)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Settings (⌘,)

struct SettingsView: View {
    @EnvironmentObject var r2: R2Client

    var body: some View {
        Form {
            Section("Voice") {
                Toggle("Speak replies", isOn: $r2.speakReplies)
                Toggle("Natural voice (on-device neural, Kokoro)", isOn: $r2.naturalVoice)
                Picker("Voice", selection: $r2.voiceName) {
                    ForEach(R2Client.voices, id: \.id) { v in Text(v.label).tag(v.id) }
                }
                .disabled(!r2.naturalVoice)
                LabeledContent("Speed") {
                    Slider(value: $r2.speechSpeed, in: 0.8...1.2, step: 0.05) {
                        EmptyView()
                    } minimumValueLabel: { Image(systemName: "tortoise") } maximumValueLabel: {
                        Image(systemName: "hare")
                    }
                }
                HStack {
                    Spacer()
                    Button("Preview Voice") { r2.previewVoice() }
                }
                Toggle("Play sounds", isOn: $r2.playSounds)
            }
            Section("R2 Core") {
                Toggle("Start the core automatically", isOn: $r2.autoStartCore)
                LabeledContent("Project folder") {
                    HStack {
                        Text(r2.repoPath)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .foregroundStyle(.secondary)
                        Button("Choose…") { choose() }
                    }
                }
                LabeledContent("Status", value: r2.phase.rawValue.capitalized)
                LabeledContent("Model", value: r2.modelName.isEmpty ? "—" : r2.modelName)
                LabeledContent("KV cache", value: r2.kvInfo.isEmpty ? "—" : r2.kvInfo)
                HStack {
                    Button("Restart Core") { r2.restartCore() }
                    Button("Show Log") { NSWorkspace.shared.open(r2.core.logURL) }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .padding(.vertical, 8)
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Use Folder"
        panel.message = "Choose the narrow-ai-with-kv project folder (contains pyproject.toml)"
        if panel.runModal() == .OK, let url = panel.url {
            r2.repoPath = url.path
        }
    }
}
