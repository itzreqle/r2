import AVFoundation

/// Native on-device speech. Word-boundary callbacks drive the orb while R2 talks.
final class Speaker: NSObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()
    var onWord: (() -> Void)?
    var onFinish: (() -> Void)?
    var onStart: (() -> Void)?

    override init() {
        super.init()
        synth.delegate = self
    }

    private lazy var voice: AVSpeechSynthesisVoice? = {
        let en = AVSpeechSynthesisVoice.speechVoices().filter { $0.language.hasPrefix("en") }
        // Prefer premium / enhanced voices when the user has downloaded them.
        return en.max { $0.quality.rawValue < $1.quality.rawValue } ?? AVSpeechSynthesisVoice(language: "en-US")
    }()

    var isSpeaking: Bool { synth.isSpeaking }

    func speak(_ text: String) {
        stop()
        let u = AVSpeechUtterance(string: text)
        u.voice = voice
        u.rate = 0.52
        u.pitchMultiplier = 1.02
        u.preUtteranceDelay = 0.05
        synth.speak(u)
    }

    func stop() {
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didStart u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onStart?() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, willSpeakRangeOfSpeechString r: NSRange, utterance u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onWord?() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onFinish?() }
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didCancel u: AVSpeechUtterance) {
        DispatchQueue.main.async { self.onFinish?() }
    }
}
