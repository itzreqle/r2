import AVFoundation

/// Microphone capture for push-to-talk. Publishes a live level (0…1) for the orb
/// and returns the recording as an in-memory 16-bit mono WAV for the R2 core.
final class AudioCapture {
    private let engine = AVAudioEngine()
    private var samples: [Float] = []
    private var sampleRate: Double = 48_000
    private let lock = NSLock()
    private let maxSeconds = 30.0
    var onLevel: ((Float) -> Void)?
    private(set) var isRunning = false

    static func requestPermission(_ done: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: done(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { ok in DispatchQueue.main.async { done(ok) } }
        default: done(false)
        }
    }

    func start() throws {
        guard !isRunning else { return }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        sampleRate = format.sampleRate
        lock.lock(); samples.removeAll(keepingCapacity: true); lock.unlock()
        let cap = Int(sampleRate * maxSeconds)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self, let ch = buffer.floatChannelData?[0] else { return }
            let n = Int(buffer.frameLength)
            let ptr = UnsafeBufferPointer(start: ch, count: n)
            var sum: Float = 0
            for v in ptr { sum += v * v }
            let rms = (sum / Float(max(n, 1))).squareRoot()
            let db = 20 * log10(max(rms, 1e-7))
            let level = max(0, min(1, (db + 52) / 40))   // -52 dB … -12 dB → 0 … 1
            self.lock.lock()
            if self.samples.count < cap { self.samples.append(contentsOf: ptr) }
            self.lock.unlock()
            self.onLevel?(level)
        }
        engine.prepare()
        try engine.start()
        isRunning = true
    }

    /// Stops recording and returns WAV data (nil if nothing useful was captured).
    func stop() -> Data? {
        guard isRunning else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
        lock.lock(); let s = samples; samples = []; lock.unlock()
        guard s.count > Int(sampleRate * 0.25) else { return nil }
        return Self.wav(s, sampleRate: Int(sampleRate))
    }

    static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(bytes)
        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in samples.indices { pcm[i] = Int16(max(-1, min(1, samples[i])) * 32767) }
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }
}
