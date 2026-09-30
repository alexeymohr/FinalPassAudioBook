import AppKit

/// Puts a finished CSV where the operator asked, using only access the sandbox has granted. An
/// existing file is never replaced — not someone else's, not this app's own earlier report: a new
/// report gets `<name> (2).csv`, `(3)`, … instead (operator).
enum Placement {
    static let audioExtensions = ["wav", "bwf", "WAV", "BWF"]

    enum Failure: LocalizedError {
        case needsFolderAccess(URL, nameTakenInRun: Bool)
        var errorDescription: String? {
            switch self {
            case .needsFolderAccess(let folder, let inRun):
                let why = inRun ? "another file in this run already uses that CSV name in “\(folder.lastPathComponent)”"
                    : "a CSV of that name already exists in “\(folder.lastPathComponent)” (it is never replaced)"
                return why + "; allow access to that folder to save this one as “… (2).csv”"
            }
        }
    }

    /// `<name>.csv` beside `<name>.wav`: the only sibling a sandboxed app may create, because the CSV
    /// type is declared as a related item in Info.plist. Read and written through a file coordinator.
    static func besideTarget(for wav: URL) -> URL { wav.deletingPathExtension().appendingPathExtension("csv") }

    /// Is anything at all at `besideTarget(for: wav)` — a file, a folder, a link?
    static func existsBeside(_ wav: URL) -> Bool {
        let target = besideTarget(for: wav)
        let presenter = RelatedItem(primary: wav, item: target)
        NSFileCoordinator.addFilePresenter(presenter)
        defer { NSFileCoordinator.removeFilePresenter(presenter) }
        var result = false
        var err: NSError?
        NSFileCoordinator(filePresenter: presenter).coordinate(readingItemAt: target, options: [], error: &err) { url in
            result = exists(url)
        }
        return result
    }

    /// Anything at `url`, a dangling link included.
    static func exists(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)) != nil || FileManager.default.fileExists(atPath: url.path)
    }

    /// `<name>.csv` beside its WAV, as a new file. nil when something is already there (the caller
    /// then uses a numbered name).
    static func besideWAV(_ csv: URL, wav: URL) throws -> URL? {
        let target = besideTarget(for: wav)
        let presenter = RelatedItem(primary: wav, item: target)
        NSFileCoordinator.addFilePresenter(presenter)
        defer { NSFileCoordinator.removeFilePresenter(presenter) }
        let data = try Data(contentsOf: csv)
        var coordinationError: NSError?
        var writeError: Error?
        var taken = false
        NSFileCoordinator(filePresenter: presenter).coordinate(writingItemAt: target, options: [],
                                                                error: &coordinationError) { url in
            taken = !writeNew(data, to: url, failure: &writeError)
        }
        if let error = coordinationError ?? writeError { throw error }
        return taken ? nil : target
    }

    /// Into a folder the app has access to: the first of `<stem>.csv`, `<stem> (2).csv`, … where
    /// nothing exists yet, not taken in this run, and not another audio file's own CSV name there.
    static func intoFolder(_ csv: URL, folder: URL, stem: String, taken: inout Set<String>) throws -> URL {
        let fm = FileManager.default
        let data = try Data(contentsOf: csv)
        for k in 1... {
            let numbered = k == 1 ? stem : "\(stem) (\(k))"
            let target = folder.appendingPathComponent("\(numbered).csv")
            if k > 1, audioExtensions.contains(where: {
                fm.fileExists(atPath: folder.appendingPathComponent("\(numbered).\($0)").path) }) {
                continue                                           // "a (2).csv" belongs to "a (2).wav"
            }
            let key = fold(target.path)
            guard !taken.contains(key), !exists(target) else { continue }
            var failure: Error?
            if !writeNew(data, to: target, failure: &failure) { continue }   // appeared just now: the next name
            if let failure { throw failure }
            taken.insert(key)
            return target
        }
        fatalError("unreachable: the numbered names never run out")
    }

    /// Write `data` as a new file at `url` (never over anything). False when something was already
    /// there. Any other failure is set in `failure`, and a file this write created part-way is removed
    /// so no cut-off report is left (the full CSV stays kept by the app).
    private static func writeNew(_ data: Data, to url: URL, failure: inout Error?) -> Bool {
        let before = exists(url)
        do {
            try data.write(to: url, options: .withoutOverwriting)
            return true
        } catch {
            let ns = error as NSError
            if ns.code == NSFileWriteFileExistsError || (ns.domain == NSPOSIXErrorDomain && ns.code == Int(EEXIST)) {
                return false
            }
            if !before && exists(url) { try? FileManager.default.removeItem(at: url) }
            failure = error
            return true
        }
    }

    // MARK: helpers

    /// How the file system compares names: APFS ignores case and Unicode normalisation.
    static func fold(_ s: String) -> String { s.precomposedStringWithCanonicalMapping.lowercased() }

    /// The path with every symlink resolved (realpath), or the standardized path if that fails.
    static func realPath(_ url: URL) -> String {
        guard let p = realpath(url.path, nil) else { return url.standardizedFileURL.path }
        defer { free(p) }
        return String(cString: p)
    }
}

private final class RelatedItem: NSObject, NSFilePresenter, @unchecked Sendable {
    let presentedItemURL: URL?
    let primaryPresentedItemURL: URL?
    let presentedItemOperationQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1          // Apple: a presenter's queue should be serial
        return q
    }()

    init(primary: URL, item: URL) {
        primaryPresentedItemURL = primary
        presentedItemURL = item
    }
}
