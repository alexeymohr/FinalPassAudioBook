import AppKit

/// Puts a finished CSV where the operator asked, using only access the sandbox has granted,
/// and never replaces a CSV this tool did not write — or one it wrote for another WAV.
enum Placement {
    /// How "ours" is recognised (after a UTF-8 BOM): the first cell of an fpab report, or the header
    /// line of a CSV an earlier version wrote. Kept in step with output.py.
    static let reportMarker = "FinalPass AudioBook report"
    static let legacyHeader = "file,time,problem,severity,end_time,check,measures"
    static let audioExtensions = ["wav", "bwf", "WAV", "BWF"]

    enum Existing { case none, ours, foreign }

    enum Failure: LocalizedError {
        case needsFolderAccess(URL, nameTakenInRun: Bool)
        var errorDescription: String? {
            switch self {
            case .needsFolderAccess(let folder, let inRun):
                let why = inRun ? "another file in this run already uses that CSV name in “\(folder.lastPathComponent)”"
                    : "a CSV of that name in “\(folder.lastPathComponent)” is not this app's report for this WAV"
                return why + "; allow access to that folder to save it as “… (2).csv”"
            }
        }
    }

    /// `<name>.csv` beside `<name>.wav`: the only sibling a sandboxed app may create, because the CSV
    /// type is declared as a related item in Info.plist. Read and written through a file coordinator.
    static func besideTarget(for wav: URL) -> URL { wav.deletingPathExtension().appendingPathExtension("csv") }

    /// What is already at `besideTarget(for: wav)`: nothing, this app's report for `wav`, or anything else.
    static func existingBeside(_ wav: URL) -> Existing {
        let target = besideTarget(for: wav)
        let presenter = RelatedItem(primary: wav, item: target)
        NSFileCoordinator.addFilePresenter(presenter)
        defer { NSFileCoordinator.removeFilePresenter(presenter) }
        var result = Existing.none
        var err: NSError?
        NSFileCoordinator(filePresenter: presenter).coordinate(readingItemAt: target, options: [], error: &err) { url in
            result = kind(of: url, for: wav)
        }
        return result
    }

    /// Ours only when this app (or the engine) wrote it for this same WAV: recorded here unchanged
    /// since (the sandbox may let the app see a CSV beside a WAV without reading it), or readable and
    /// starting like one of ours, naming this WAV — and, away from the WAV's folder, its folder too.
    /// A symlink is never ours: nothing is written through one.
    static func kind(of url: URL, for wav: URL) -> Existing {
        let fm = FileManager.default
        if let type = try? fm.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType, type == .typeSymbolicLink {
            return .foreign
        }
        guard fm.fileExists(atPath: url.path) else { return .none }
        if let sig = signature(url), let record = written()[url.path] {
            let parts = record.split(separator: "\t", maxSplits: 1).map(String.init)
            if parts.first == sig {
                return parts.count < 2 || sameFile(parts[1], wav.path) ? .ours : .foreign
            }
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else { return .foreign }
        defer { try? handle.close() }
        var text = String(decoding: (try? handle.read(upToCount: 8192)) ?? Data(), as: UTF8.self)
        if text.hasPrefix("\u{FEFF}") { text.removeFirst() }
        let rows = csvRows(text, limit: 40)
        guard let first = rows.first else { return .foreign }
        if first.joined(separator: ",") == legacyHeader { return .ours }
        guard first.first == reportMarker else { return .foreign }
        var facts: [String: String] = [:]
        for row in rows.dropFirst() {
            if row.allSatisfy({ $0.isEmpty }) { break }                // the summary ends at the first empty row
            if facts[row[0]] == nil { facts[row[0]] = row.count > 1 ? uncell(row[1]) : "" }
        }
        if let name = facts["file"], !name.isEmpty, fold(name) != fold(wav.lastPathComponent) { return .foreign }
        let here = realPath(wav.deletingLastPathComponent())
        if let folder = facts["folder"], !folder.isEmpty,
           fold(realPath(url.deletingLastPathComponent())) != fold(here), fold(folder) != fold(here) {
            return .foreign
        }
        return .ours
    }

    // MARK: CSVs this app wrote: path -> size and modification time when written, and for which WAV

    private static let writtenKey = "writtenCSVs"

    private static func written() -> [String: String] {
        UserDefaults.standard.dictionary(forKey: writtenKey) as? [String: String] ?? [:]
    }

    private static func signature(_ url: URL) -> String? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = a[.size] as? NSNumber, let date = a[.modificationDate] as? Date else { return nil }
        return "\(size.int64Value):\(Int64(date.timeIntervalSince1970 * 1000))"
    }

    static func remember(_ url: URL, wav: URL) {
        var d = written()
        if d.count > 5000 { d = [:] }               // a bounded record; forgetting only means "(2)" names
        if let sig = signature(url) { d[url.path] = "\(sig)\t\(wav.path)" }
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
            // In place (the sandbox allows no temporary sibling). A write that fails part-way must not
            // leave a cut-off report that reads as ours: remove it; the full CSV is kept by the app.
            do { try data.write(to: url) } catch {
                writeError = error
                try? FileManager.default.removeItem(at: url)
            }
        }
        if let error = coordinationError ?? writeError { throw error }
        remember(target, wav: wav)
        return target
    }

    /// Into a folder the app has access to: `<stem>.csv`, or `<stem> (2).csv`, … when that name is
    /// taken in this run, held by anything but this app's report for `wav`, or is another audio file's
    /// own CSV name there. Our own earlier CSV for the same WAV is replaced.
    static func intoFolder(_ csv: URL, folder: URL, stem: String, wav: URL, taken: inout Set<String>) throws -> URL {
        let fm = FileManager.default
        for k in 1... {
            let numbered = k == 1 ? stem : "\(stem) (\(k))"
            let target = folder.appendingPathComponent("\(numbered).csv")
            if k > 1, audioExtensions.contains(where: {
                fm.fileExists(atPath: folder.appendingPathComponent("\(numbered).\($0)").path) }) {
                continue                                           // "a (2).csv" belongs to "a (2).wav"
            }
            let key = fold(target.path)
            guard !taken.contains(key), kind(of: target, for: wav) != .foreign else { continue }
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: target.path, isDirectory: &isDir), isDir.boolValue { continue }
            let data = try Data(contentsOf: csv)
            try data.write(to: target, options: .atomic)
            taken.insert(key)
            remember(target, wav: wav)
            return target
        }
        fatalError("unreachable: the numbered names never run out")
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

    private static func sameFile(_ a: String, _ b: String) -> Bool {
        fold(realPath(URL(fileURLWithPath: a))) == fold(realPath(URL(fileURLWithPath: b)))
    }

    /// A cell as the engine wrote it: a leading ' only guards a formula-looking value.
    private static func uncell(_ s: String) -> String {
        guard s.hasPrefix("'"), let next = s.dropFirst().first, "=+-@\t\r".contains(next) else { return s }
        return String(s.dropFirst())
    }

    /// The first `limit` rows of CSV text (RFC 4180 quoting; a row cut off by the read is dropped).
    static func csvRows(_ text: String, limit: Int) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var cell = ""
        var quoted = false
        let chars = Array(text)
        var i = 0
        while i < chars.count && rows.count < limit {
            let c = chars[i]
            if quoted {
                if c == "\"" {
                    if i + 1 < chars.count && chars[i + 1] == "\"" { cell.append("\""); i += 1 } else { quoted = false }
                } else {
                    cell.append(c)
                }
            } else if c == "\"" {
                quoted = true
            } else if c == "," {
                row.append(cell)
                cell = ""
            } else if c == "\n" || c == "\r\n" || c == "\r" {
                row.append(cell)
                rows.append(row)
                row = []
                cell = ""
            } else {
                cell.append(c)
            }
            i += 1
        }
        return rows
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
