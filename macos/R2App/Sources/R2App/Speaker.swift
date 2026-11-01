import AVFoundation

/// R2's voice.
///
/// * Natural mode: fetches a neural Kokoro voice from the core (`POST /tts`), plays it
///   with AVAudioPlayer and meters the *actual* loudness 60×/s to drive the orb.
/// * Fallback: the system voice (AVSpeechSynthesizer), with word-boundary pulses.
final class Speaker: NSObject, AVSpeechSynthesizerDelegate, AVAudioPlayerDelegate {
    private let synth = AVSpeechSynthesizer()
    private var player: AVAudioPlayer?
    private var meterTimer: Timer?
    private var fetchTask: URLSessionDataTask?

    var onStart: (() -> Void)?
    var onWord: (() -> Void)?
    var onLevel: ((Float) -> Void)?
    var onFinish: (() -> Void)?

    override init() {
        super.init()
        synth.delegate = self
    }

    private lazy var systemVoice: AVSpeechSynthesisVoice? = {
        let en = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("en") }
        return en.max { $0.quality.rawValue < $1.quality.rawValue } ?? AVSpeechSynthesisVoice(language: "en-US")
    }()

    var isSpeaking: Bool { synth.isSpeaking || (player?.isPlaying ?? false) }

    func speak(_ text: String, natural: Bool, voice: String, speed: Double, server: URL, token: String?) {
        stop()
        guard natural, let token, !token.isEmpty else { return speakSystem(text, speed: speed) }

        var req = URLRequest(url: server.appendingPathComponent("tts"))
        req.httpMethod = "POST"
        req.timeoutInterval = 20
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["text": text, "voice": voice, "speed": speed])
        let task = URLSession.shared.dataTask(with: req) { [weak self] data, resp, _ in
            DispatchQueue.main.async {
                guard let self else { return }
                let ok = (resp as? HTTPURLResponse)?.statusCode == 200
                if ok, let data, self.play(data) { return }
                self.speakSystem(text, speed: speed)   // fall back, never stay silent
            }
        }
        fetchTask = task
        task.resume()
    }

    private func play(_ data: Data) -> Bool {
        guard let p = try? AVAudioPlayer(data: data) else { return false }
        p.delegate = self
        p.isMeteringEnabled = true
        p.prepareToPlay()
        guard p.play() else { return false }
        player = p
        onStart?()
        meterTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self, let p = self.player, p.isPlaying else { return }
            p.updateMeters()
            let db = p.averagePower(forChannel: 0)            // -160 … 0 dB
            self.onLevel?(max(0, min(1, (db + 42) / 36)))     // speech sits ~ -40 … -6 dB
        }
        return true
    }

    private func speakSystem(_ text: String, speed: Double) {
        let u = AVSpeechUtterance(string: text)
        u.voice = systemVoice
        u.rate = Float(0.5 * speed)
        u.pitchMultiplier = 1.0
        u.preUtteranceDelay = 0.03
        synth.speak(u)
    }

    func stop() {
        fetchTask?.cancel()
        fetchTask = nil
        meterTimer?.invalidate()
        meterTimer = nil
        if let p = player, p.isPlaying { p.stop() }
        player = nil
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
        onLevel?(0)
    }

    private func finished() {
        meterTimer?.invalidate()
        meterTimer = nil
        player = nil
        onLevel?(0)
        onFinish?()
    }

    // MARK: AVAudioPlayerDelegate
    func audioPlayerDidFinishPlaying(_ p: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { self.finished() }
    }

    // MARK: AVSpeechSynthesizerDelegate
    func speechSynthesizer(_ s: AVSpeechSynthesizer, didStart u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onStart?() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, willSpeakRangeOfSpeechString r: NSRange, utterance u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onWord?() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.finished() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didCancel u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onFinish?() }
    }
}
