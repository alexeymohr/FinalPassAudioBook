import Foundation

/// One line of `fpab check --progress jsonl`.
struct EngineEvent: Decodable, Sendable {
    let event: String
    let index: Int?
    let total: Int?
    let files: Int?
    let stage: String?
    let step: Int?
    let steps: Int?
    let path: String?
    let paths: [String]?
    let csv: String?
    let notes: [String]?
    let sev3: Int?
    let sev2: Int?
    let sev1: Int?
    let network_attempts: Int?
}

/// The bundled engine: a self-contained Python with fpab, run inside this app's sandbox.
enum Engine {
    static var resources: URL { Bundle.main.resourceURL!.appendingPathComponent("engine") }
    static var python: URL { resources.appendingPathComponent("python/bin/python3.12") }
    static var modelRoot: URL { resources.appendingPathComponent("model") }

    static var isInstalled: Bool { FileManager.default.isExecutableFile(atPath: python.path) }

    /// `fpab check` for these files, writing one CSV per file into `csvDir` (a folder in the app's container).
    static func process(files: [URL], csvDir: URL, withPauses: Bool) -> Process {
        let p = Process()
        p.executableURL = python
        // -I: ignore the environment's Python settings; -B: never write .pyc into the signed bundle.
        var args = ["-I", "-B", "-c", "import sys; sys.argv[0] = 'fpab'; from finalpass_audiobook.cli import main; main()",
                    "check", "--csv-dir", csvDir.path, "--progress", "jsonl"]
        if withPauses { args.append("--with-pauses") }
        p.arguments = args + ["--"] + files.map(\.path)
        var env = ProcessInfo.processInfo.environment.filter { !$0.key.hasPrefix("DYLD_") && !$0.key.hasPrefix("PYTHON") }
        env["FPAB_MODEL_DIR"] = modelRoot.path
        p.environment = env
        return p
    }
}

/// Waits for a process to exit without blocking a thread: the termination handler resumes the waiter.
final class ExitWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var status: (Int32, Process.TerminationReason)?
    private var waiter: CheckedContinuation<(Int32, Process.TerminationReason), Never>?

    /// Install before `run()`, so an early exit cannot be missed.
    func attach(to process: Process) {
        process.terminationHandler = { [weak self] p in self?.finish((p.terminationStatus, p.terminationReason)) }
    }

    private func finish(_ value: (Int32, Process.TerminationReason)) {
        lock.lock()
        status = value
        let w = waiter
        waiter = nil
        lock.unlock()
        w?.resume(returning: value)
    }

    func wait() async -> (Int32, Process.TerminationReason) {
        await withCheckedContinuation { c in
            lock.lock()
            if let s = status {
                lock.unlock()
                c.resume(returning: s)
            } else {
                waiter = c
                lock.unlock()
            }
        }
    }
}
