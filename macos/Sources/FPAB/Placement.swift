import AppKit

/// Puts a finished CSV where the operator asked, using only access the sandbox has granted,
/// and never replaces a CSV this tool did not write.
enum Placement {
    /// The header every fpab CSV starts with (after a UTF-8 BOM): how "ours" is recognised.
    static let header = "file,time,problem,severity,end_time,check,measures"

    enum Existing { case none, ours, foreign }

    enum Failure: LocalizedError {
        case needsFolderAccess(URL, nameTakenInRun: Bool)
        var errorDescription: String? {
            switch self {
            case .needsFolderAccess(let folder, let inRun):
                let why = inRun ? "another file in this run already uses that CSV name in “\(folder.lastPathComponent)”"
                    : "a CSV of that name in “\(folder.lastPathComponent)” is not one this app wrote"
                return why + "; allow access to that folder to save it as “… (2).csv”"
            }
        }
    }

    /// `<name>.csv` beside `<name>.wav`: the only sibling a sandboxed app may create, because the CSV
    /// type is declared as a related item in Info.plist. Read and written through a file coordinator.
    static func besideTarget(for wav: URL) -> URL { wav.deletingPathExtension().appendingPathExtension("csv") }

    /// What is already at `besideTarget(for: wav)`: nothing, a CSV we wrote, or someone else's file.
    static func existingBeside(_ wav: URL) -> Existing {
        let target = besideTarget(for: wav)
        let presenter = RelatedItem(primary: wav, item: target)
        NSFileCoordinator.addFilePresenter(presenter)
        defer { NSFileCoordinator.removeFilePresenter(presenter) }
        var result = Existing.none
        var err: NSError?
        NSFileCoordinator(filePresenter: presenter).coordinate(readingItemAt: target, options: [], error: &err) { url in
            result = kind(of: url)
        }
        return result
    }

    /// Ours if this app wrote it and it is unchanged since (the sandbox may let the app see a CSV
    /// beside a WAV without reading it), or if it is readable and starts with our header.
    static func kind(of url: URL) -> Existing {
        guard FileManager.default.fileExists(atPath: url.path) else { return .none }
        if let sig = signature(url), written()[url.path] == sig { return .ours }
        guard let handle = try? FileHandle(forReadingFrom: url) else { return .foreign }
        defer { try? handle.close() }
        let head = String(decoding: (try? handle.read(upToCount: 256)) ?? Data(), as: UTF8.self)
        let first = head.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? ""
        let clean = first.trimmingCharacters(in: CharacterSet(charactersIn: "\u{FEFF}\r"))
        return clean == header ? .ours : .foreign
    }

    // MARK: CSVs this app wrote: path -> size and modification time when written

    private static let writtenKey = "writtenCSVs"

    private static func written() -> [String: String] {
        UserDefaults.standard.dictionary(forKey: writtenKey) as? [String: String] ?? [:]
    }

    private static func signature(_ url: URL) -> String? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = a[.size] as? NSNumber, let date = a[.modificationDate] as? Date else { return nil }
        return "\(size.int64Value):\(Int64(date.timeIntervalSince1970 * 1000))"
    }

    static func remember(_ url: URL) {
        var d = written()
        if d.count > 5000 { d = [:] }               // a bounded record; forgetting only means "(2)" names
        d[url.path] = signature(url)
        UserDefaults.standard.set(d, forKey: writtenKey)
    }

    static func besideWAV(_ csv: URL, wav: URL) throws -> URL {
        let target = besideTarget(for: wav)
        let presenter = RelatedItem(primary: wav, item: target)
        NSFileCoordinator.addFilePresenter(presenter)
        defer { NSFileCoordinator.removeFilePresenter(presenter) }
        let data = try Data(contentsOf: csv)
        var coordinationError: NSError?
        var writeError: Error?
        NSFileCoordinator(filePresenter: presenter).coordinate(writingItemAt: target, options: .forReplacing,
                                                                error: &coordinationError) { url in
            do { try data.write(to: url) } catch { writeError = error }
        }
        if let error = coordinationError ?? writeError { throw error }
        remember(target)
        return target
    }

    /// Into a folder the app has access to: `<stem>.csv`, or `<stem> (2).csv`, … when that name is
    /// taken in this run or held by a file this app did not write. Our own earlier CSV is replaced.
    static func intoFolder(_ csv: URL, folder: URL, stem: String, taken: inout Set<String>) throws -> URL {
        var k = 1
        while true {
            let name = k == 1 ? "\(stem).csv" : "\(stem) (\(k)).csv"
            let target = folder.appendingPathComponent(name)
            let key = target.path.precomposedStringWithCanonicalMapping.lowercased()
            let existing = kind(of: target)
            if !taken.contains(key) && existing != .foreign {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: target.path, isDirectory: &isDir), isDir.boolValue {
                    k += 1
                    continue
                }
                let data = try Data(contentsOf: csv)
                try data.write(to: target, options: .atomic)
                taken.insert(key)
                remember(target)
                return target
            }
            k += 1
        }
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
