import AppKit
import SwiftUI

// MARK: - Multi-step run

/// Checklist for a multi-step run ("check git status then find the swift files").
/// Display only: confirming/cancelling uses the normal Confirm/Cancel bar and voice "yes".
struct RunStepsCard: View {
    let steps: [R2Client.RunStep]
    let status: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("\(steps.count) steps", systemImage: "list.number")
                Spacer()
                Text(statusLabel).foregroundStyle(statusColor)
            }
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .foregroundStyle(.white.opacity(0.6))
            ForEach(steps) { step in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Image(systemName: icon(step.status))
                        .font(.system(size: 16))
                        .foregroundStyle(color(step.status))
                        .frame(width: 20)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(step.title.capitalized)
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white.opacity(step.status == "skipped" ? 0.4 : 0.95))
                        if !step.speech.isEmpty && step.status != "pending" && step.status != "skipped" {
                            Text(step.speech)
                                .font(.system(size: 14, design: .rounded))
                                .foregroundStyle(.white.opacity(0.72))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Step \(step.id + 1), \(step.title), \(step.status.replacingOccurrences(of: "_", with: " "))")
            }
        }
        .padding(16)
        .fixedSize(horizontal: false, vertical: true)   // never squeezed by the window height
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.06))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08)))
        )
        .padding(.horizontal, 24)
        .transition(.opacity.combined(with: .move(edge: .bottom)))
    }

    private var statusLabel: String {
        switch status {
        case "awaiting_confirmation": return "Waiting for you"
        case "completed": return "Done"
        case "failed": return "Stopped"
        case "rejected": return "Not started"
        case "cancelled": return "Cancelled"
        default: return "Running"
        }
    }

    private var statusColor: Color {
        switch status {
        case "completed": return .green
        case "awaiting_confirmation": return .orange
        case "failed", "rejected": return .red
        default: return .white.opacity(0.55)
        }
    }

    private func icon(_ s: String) -> String {
        switch s {
        case "ok": return "checkmark.circle.fill"
        case "failed": return "xmark.circle.fill"
        case "awaiting_confirmation": return "questionmark.circle.fill"
        case "running": return "circle.dotted"
        case "skipped": return "minus.circle"
        case "cancelled": return "slash.circle"
        default: return "circle"
        }
    }

    private func color(_ s: String) -> Color {
        switch s {
        case "ok": return .green
        case "failed": return .red
        case "awaiting_confirmation": return .orange
        default: return .white.opacity(0.4)
        }
    }
}

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
                Spacer(minLength: runMode ? 8 : 30)

                // A multi-step run needs room for its step list: the orb steps back.
                OrbView(phase: r2.phase, level: r2.level, speechTick: r2.speechTick)
                    .frame(width: orbSize, height: orbSize)   // room for the halo to fade out
                    .padding(.vertical, -orbSize * 0.15)
                    .contentShape(Circle().inset(by: orbSize * 0.275))
                    .onTapGesture { r2.toggleListening() }
                    .help(r2.phase == .listening ? "Click to stop" : "Click to talk (⌘L)")

                if runMode {
                    // Scroll instead of squeezing: every card keeps its natural height.
                    ScrollView(.vertical) {
                        conversation.padding(.bottom, 8)
                    }
                    .scrollIndicators(.automatic)
                } else {
                    conversation
                    Spacer(minLength: 16)
                }

                inputBar
                footer
            }
            .padding(.bottom, 14)
        }
        .preferredColorScheme(.dark)
        .animation(.smooth(duration: 0.35), value: r2.phase)
        .animation(.smooth(duration: 0.35), value: r2.detail)
        .animation(.smooth(duration: 0.35), value: r2.runSteps)
        .onAppear { inputFocused = true }
    }

    private var runMode: Bool { !r2.runSteps.isEmpty }
    private var orbSize: CGFloat { runMode ? 220 : 400 }

    /// Caption, reply, run steps, confirmation and output, top to bottom.
    @ViewBuilder private var conversation: some View {
        captionView
            .padding(.top, 4)

        replyView
            .padding(.top, 10)

        if runMode { RunStepsCard(steps: r2.runSteps, status: r2.runStatus).padding(.top, 14) }

        if r2.phase == .confirm { confirmBar.padding(.top, 18) }

        // While a run waits for a yes, the step list says it all; show output after.
        if !r2.detail.isEmpty && !(r2.phase == .confirm && runMode) {
            detailCard.padding(.top, 16)
        }

        // Ideas only before the first request (and after New Conversation).
        if r2.detail.isEmpty && r2.caption.isEmpty && (r2.phase == .idle || r2.phase == .offline) {
            SuggestionChips()
                .padding(.top, 22)
                .transition(.opacity)
        }
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
            .font(.system(size: replySize, weight: .semibold, design: .rounded))
            .foregroundStyle(
                LinearGradient(colors: [.white, .white.opacity(0.82)], startPoint: .top, endPoint: .bottom))
            .multilineTextAlignment(.center)
            .lineLimit(8)
            .minimumScaleFactor(0.8)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 36)
            .textSelection(.enabled)
            .id("reply-\(r2.reply)")
            .transition(.asymmetric(
                insertion: .opacity.combined(with: .offset(y: 8)).combined(with: .scale(scale: 0.98)),
                removal: .opacity))
    }

    /// Short replies are big like Siri's; spoken answers get smaller so they fit in full.
    private var replySize: CGFloat {
        switch r2.reply.count {
        case ..<70: return 26
        case ..<150: return 21
        default: return 18
        }
    }

    private var confirmBar: some View {
        HStack(spacing: 12) {
            Button("Cancel", role: .cancel) { r2.cancel() }
                .keyboardShortcut(.cancelAction)
            // ⌘Return, not plain Return: a stray Enter after R2 grabs focus (⌥Space,
            // launch) must never approve a privileged action.
            Button("Confirm") { r2.confirm() }
                .keyboardShortcut(.return, modifiers: .command)
                .help("Confirm (⌘Return)")
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
                .frame(minHeight: 60, maxHeight: 170)
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
    @State private var openAtLogin = LoginItem.isEnabled
    @State private var loginError = ""

    var body: some View {
        Form {
            Section("General") {
                Toggle(isOn: $r2.hotKeyEnabled) {
                    Text("Summon R2 with ⌥Space")
                    Text("Brings R2 forward and starts listening; press again to stop.")
                }
                if !r2.hotKeyError.isEmpty {
                    Label(r2.hotKeyError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
                Toggle("Open at login", isOn: $openAtLogin)
                    .onChange(of: openAtLogin) { _, on in
                        loginError = LoginItem.set(on) ?? ""
                        openAtLogin = LoginItem.isEnabled
                    }
                if !loginError.isEmpty {
                    Label(loginError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
            }
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
