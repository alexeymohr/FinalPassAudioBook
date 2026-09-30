import AppKit
import Observation
import UniformTypeIdentifiers

/// The list of WAVs, the two settings, and one run of the engine at a time.
@MainActor @Observable
final class RunModel {
    enum State: Equatable { case waiting, running, done, unsaved(String), failed(String), cancelled }

    struct Item: Identifiable {
        let id = UUID()
        let url: URL
        var state: State = .waiting
        var csv: URL?
        var summary = ""
        var notes: [String] = []
    }

    enum Output: String { case besideWAV, folder }

    static let audioExtensions: Set<String> = ["wav", "bwf"]

    var items: [Item] = []
    var running = false
    var progress = 0.0
    var status = "Drop WAV files to begin."
    var errorText: String?
    var runNotes: [String] = []

    var output: Output = Output(rawValue: UserDefaults.standard.string(forKey: "output") ?? "") ?? .besideWAV {
        didSet { UserDefaults.standard.set(output.rawValue, forKey: "output") }
    }
    var withPauses = UserDefaults.standard.bool(forKey: "withPauses") {
        didSet { UserDefaults.standard.set(withPauses, forKey: "withPauses") }
    }
    private(set) var folder: URL? = RunModel.resolveBookmark(UserDefaults.standard.data(forKey: "folderBookmark"),
                                                              key: "folderBookmark")

    var unsavedCount: Int { items.filter { if case .unsaved = $0.state { true } else { false } }.count }
    var canGo: Bool { !running && !items.isEmpty && Engine.isInstalled && (output == .besideWAV || folder != nil) }

    @ObservationIgnored private var process: Process?
    @ObservationIgnored private var workDir: URL?
    @ObservationIgnored private var stderrFile: URL?
    @ObservationIgnored private var fileIndex = 0
    @ObservationIgnored private var cancelled = false
    @ObservationIgnored private var quitWhenDone = false
    @ObservationIgnored private var taken = Set<String>()          // CSV paths used in this run (case-folded)
    @ObservationIgnored private var altFolders: [String: URL] = [:] // folder -> granted access, for "(2)" names

    init() { Self.sweepOldRuns() }

    // MARK: files

    func add(_ urls: [URL]) {
        guard !running else { return }
        let fm = FileManager.default
        var found: [URL] = []
        var emptyFolders: [String] = []
        var links = 0
        for url in urls {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
            if values?.isSymbolicLink == true {
                links += 1                          // the sandbox grants the link, not what it points to
            } else if values?.isDirectory == true {
                let inside = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isRegularFileKey],
                                                          options: [.skipsHiddenFiles])) ?? []
                let wavs = inside.filter { Self.isAudioFile($0) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
                if wavs.isEmpty { emptyFolders.append(url.lastPathComponent) }
                found += wavs
            } else if Self.isAudioFile(url) {
                found.append(url)
            }
        }
        var known = Set(items.map { Self.identity($0.url) })
        for url in found where known.insert(Self.identity(url)).inserted {         // each file once
            items.append(Item(url: url))
        }
        if links > 0 {
            status = "\(links) dropped item\(links == 1 ? " is a link" : "s are links"); drop the original files instead."
        } else if !emptyFolders.isEmpty {
            status = "No WAV files found in “\(emptyFolders.joined(separator: "”, “"))” (sub-folders are not searched)."
        } else if !items.isEmpty {
            status = "\(items.count) file\(items.count == 1 ? "" : "s") ready."
        }
    }

    /// The same file by any path (device and file number), or its resolved path if it cannot be looked at.
    private static func identity(_ url: URL) -> String {
        if let a = try? FileManager.default.attributesOfItem(atPath: url.path),
           let dev = a[.systemNumber] as? NSNumber, let ino = a[.systemFileNumber] as? NSNumber {
            return "\(dev):\(ino)"
        }
        return Placement.realPath(url)
    }

    /// A regular file (not a folder, even one named "x.wav") with a WAV/BWF extension.
    private static func isAudioFile(_ url: URL) -> Bool {
        audioExtensions.contains(url.pathExtension.lowercased())
            && (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
    }

    func remove(_ id: UUID) {
        guard !running, let i = items.firstIndex(where: { $0.id == id }) else { return }
        if case .unsaved = items[i].state {
            guard confirmDiscard(1, then: "Remove") else { return }
            discardKept(items[i])
        }
        items.remove(at: i)
    }

    func clear() {
        guard !running, confirmDiscard(unsavedCount, then: "Clear") else { return }
        items.forEach(discardKept)
        items.removeAll()
        progress = 0
        status = "Drop WAV files to begin."
        errorText = nil
        runNotes = []
    }

    func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.wav] + [UTType(filenameExtension: "bwf")].compactMap { $0 }
        if panel.runModal() == .OK { add(panel.urls) }
    }

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Use Folder"
        guard panel.runModal() == .OK, let url = panel.url,
              let bookmark = try? url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil,
                                                   relativeTo: nil) else { return }
        UserDefaults.standard.set(bookmark, forKey: "folderBookmark")
        folder = url
        output = .folder
    }

    /// A saved security-scoped bookmark, refreshed when macOS reports it stale. Never waits on a
    /// server or asks to mount one (an offline network folder reads as unreachable).
    private static func resolveBookmark(_ data: Data?, key: String?) -> URL? {
        guard let data else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: [.withSecurityScope, .withoutUI, .withoutMounting],
                                 relativeTo: nil, bookmarkDataIsStale: &stale) else { return nil }
        if stale, let key, url.startAccessingSecurityScopedResource() {
            defer { url.stopAccessingSecurityScopedResource() }
            if let fresh = try? url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil,
                                                 relativeTo: nil) {
                UserDefaults.standard.set(fresh, forKey: key)
            }
        }
        return url
    }

    /// Access to a WAV's own folder, for writing "<name> (2).csv" beside it (the sandbox allows only
    /// "<name>.csv" otherwise). Remembered per folder; asked for once when needed.
    private func folderAccess(for dir: URL, ask: Bool) -> URL? {
        var saved = UserDefaults.standard.dictionary(forKey: "folderAccess") as? [String: Data] ?? [:]
        if let url = Self.resolveBookmark(saved[dir.path], key: nil) { return url }
        guard ask else { return nil }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.directoryURL = dir
        panel.prompt = "Allow"
        panel.message = "A CSV you made already has the name this app would use in “\(dir.lastPathComponent)”. "
            + "Allow this folder so the report can be saved as “… (2).csv” instead of replacing yours."
        guard panel.runModal() == .OK, let url = panel.url,
              url.standardizedFileURL.path == dir.standardizedFileURL.path,
              let bookmark = try? url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil,
                                                   relativeTo: nil) else { return nil }
        saved[dir.path] = bookmark
        UserDefaults.standard.set(saved, forKey: "folderAccess")
        return url
    }

    // MARK: run

    func go() {
        guard canGo, quitWhenDone || confirmDiscard(unsavedCount, then: "Go") else { return }
        errorText = nil
        runNotes = []
        taken = []
        altFolders = [:]
        if output == .folder {
            guard let folder, folder.startAccessingSecurityScopedResource() else {
                self.folder = nil
                errorText = "The chosen folder can no longer be reached. Choose it again."
                return
            }
            folder.stopAccessingSecurityScopedResource()
        } else {
            preflightBesideWAVs()
        }
        let fm = FileManager.default
        let work = fm.temporaryDirectory.appendingPathComponent("\(Self.runPrefix)\(UUID().uuidString)", isDirectory: true)
        let errFile = work.appendingPathComponent("engine-stderr.txt")
        do {
            try fm.createDirectory(at: work, withIntermediateDirectories: true)
            fm.createFile(atPath: errFile.path, contents: nil)
            try "\(getpid())".write(to: work.appendingPathComponent(Self.ownerFile), atomically: true, encoding: .utf8)
        } catch {
            errorText = "Could not create a working folder: \(error.localizedDescription)"
            return
        }
        items.forEach(discardKept)                  // agreed above; the run is about to start
        for i in items.indices {
            items[i].state = .waiting
            items[i].csv = nil
            items[i].summary = ""
            items[i].notes = []
        }
        workDir = work
        stderrFile = errFile
        cancelled = false
        progress = 0
        fileIndex = 0
        status = "Starting…"

        let p = Engine.process(files: items.map(\.url), csvDir: work, withPauses: withPauses)
        let out = Pipe()
        p.standardOutput = out
        // stderr goes to a file: two pipes read at once can block each other (and the engine)
        p.standardError = try? FileHandle(forWritingTo: errFile)
        let exit = ExitWaiter()
        exit.attach(to: p)
        do { try p.run() } catch {
            errorText = "Could not start the engine: \(error.localizedDescription)"
            try? fm.removeItem(at: work)
            workDir = nil
            endAutorunIfNeeded()
            return
        }
        running = true
        process = p
        let stdout = out.fileHandleForReading
        Task.detached { [weak self] in
            do {
                for try await line in stdout.bytes.lines {
                    guard let data = line.data(using: .utf8),
                          let event = try? JSONDecoder().decode(EngineEvent.self, from: data) else { continue }
                    await self?.handle(event)
                }
            } catch {}
            let (code, reason) = await exit.wait()      // every progress line is handled before cleanup
            await self?.finished(code, reason)
        }
    }

    /// Before the run: where "<name>.csv" beside a WAV would replace someone else's file, or two WAVs
    /// in one folder would share it, ask once for that folder so "<name> (2).csv" can be written.
    private func preflightBesideWAVs() {
        var seen = Set<String>()
        var needFolders: [URL] = []
        for item in items {
            let key = Self.key(Placement.besideTarget(for: item.url))
            let clash = seen.contains(key) || Placement.existingBeside(item.url) == .foreign
            seen.insert(key)
            let dir = item.url.deletingLastPathComponent()
            if clash && !needFolders.contains(where: { $0.path == dir.path }) { needFolders.append(dir) }
        }
        for dir in needFolders {
            if let url = folderAccess(for: dir, ask: !quitWhenDone) { altFolders[dir.path] = url }
        }
    }

    private static func key(_ url: URL) -> String { url.path.precomposedStringWithCanonicalMapping.lowercased() }

    #if FPAB_TEST_HOOKS
    /// Go once the files are in, and quit when the run ends (scripted checks of a test build only).
    func autorun() {
        quitWhenDone = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self else { return }
            self.go()
            if !self.running { self.endAutorunIfNeeded() }
        }
    }
    #endif

    var isRunning: Bool { process?.isRunning ?? false }

    func cancel() {
        guard let p = process, p.isRunning else { return }
        cancelled = true
        p.terminate()
    }

    /// Quit while running: stop the engine and remove the working folder.
    func stopForQuit() {
        cancel()
        process?.waitUntilExit()
        if let workDir { try? FileManager.default.removeItem(at: workDir) }
    }

    private func handle(_ e: EngineEvent) {
        let n = max(items.count, 1)
        switch e.event {
        case "start":
            if let paths = e.paths, paths != items.map(\.url.path) {
                errorText = "The engine received a different file list than the app shows; stopped."
                cancel()
            }
        case "file":
            fileIndex = e.index ?? fileIndex
            guard items.indices.contains(fileIndex), e.path == nil || e.path == items[fileIndex].url.path else { return }
            items[fileIndex].state = .running
            status = "File \(fileIndex + 1) of \(n): \(items[fileIndex].url.lastPathComponent)"
        case "stage":
            if (e.index ?? 0) < 0 {
                status = "Loading the chopped-word model…"
            } else if let step = e.step, let steps = e.steps, steps > 0, step >= 0 {
                progress = min(1, (Double(fileIndex) + Double(step + 1) / Double(steps)) / Double(n))
                status = "File \(fileIndex + 1) of \(n): \(items[safe: fileIndex]?.url.lastPathComponent ?? "") — \(e.stage ?? "")"
            }
        case "file_done":
            guard let i = e.index, items.indices.contains(i) else { return }
            progress = min(1, Double(i + 1) / Double(n))
            items[i].notes = e.notes ?? []
            guard e.path == nil || e.path == items[i].url.path else {
                items[i].state = .failed("the result did not match this file")
                return
            }
            guard let csvPath = e.csv else {
                let why = (e.notes ?? []).joined(separator: "; ")
                items[i].state = .failed(why.isEmpty ? "skipped (no reason given)" : why)
                return
            }
            let csv = URL(fileURLWithPath: csvPath)
            items[i].summary = "\(e.sev3 ?? 0) × sev 3 · \(e.sev2 ?? 0) × sev 2 · \(e.sev1 ?? 0) × sev 1"
            do {
                items[i].csv = try place(csv, wav: items[i].url)
                items[i].state = .done
            } catch {
                if let kept = keepUnsaved(csv) {
                    items[i].csv = kept
                    items[i].state = .unsaved(error.localizedDescription)
                } else {
                    items[i].state = .failed("not saved, and the report could not be kept: \(error.localizedDescription)")
                }
            }
        case "done":
            runNotes = e.notes ?? []
            status = "Done. Network attempts: \(e.network_attempts ?? 0) (macOS blocks the network for this app)."
        default:
            break
        }
    }

    private func place(_ csv: URL, wav: URL) throws -> URL {
        let stem = wav.deletingPathExtension().lastPathComponent
        switch output {
        case .folder:
            guard let folder else { throw CocoaError(.fileNoSuchFile) }
            let granted = folder.startAccessingSecurityScopedResource()
            defer { if granted { folder.stopAccessingSecurityScopedResource() } }
            guard granted else { throw CocoaError(.fileWriteNoPermission) }
            return try Placement.intoFolder(csv, folder: folder, stem: stem, wav: wav, taken: &taken)
        case .besideWAV:
            let key = Self.key(Placement.besideTarget(for: wav))
            let inRun = taken.contains(key)
            if !inRun && Placement.existingBeside(wav) != .foreign {
                let placed = try Placement.besideWAV(csv, wav: wav)
                taken.insert(key)
                return placed
            }
            let dir = wav.deletingLastPathComponent()
            guard let alt = altFolders[dir.path] else { throw Placement.Failure.needsFolderAccess(dir, nameTakenInRun: inRun) }
            let granted = alt.startAccessingSecurityScopedResource()
            defer { if granted { alt.stopAccessingSecurityScopedResource() } }
            return try Placement.intoFolder(csv, folder: alt, stem: stem, wav: wav, taken: &taken)
        }
    }

    /// Where CSVs that could not be put in place wait to be saved: in the app's own storage, one
    /// folder each (the name stays as it was). Any left when the app quits (or stops) are offered
    /// again at the next launch.
    private static var unsavedRoot: URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Unsaved Reports", isDirectory: true)
    }

    /// A CSV that could not be put in place is kept, never thrown away without asking.
    private func keepUnsaved(_ csv: URL) -> URL? {
        let fm = FileManager.default
        guard let dir = Self.unsavedRoot?.appendingPathComponent(UUID().uuidString, isDirectory: true),
              (try? fm.createDirectory(at: dir, withIntermediateDirectories: true)) != nil else { return nil }
        try? "\(getpid())".write(to: dir.appendingPathComponent(Self.ownerFile), atomically: true, encoding: .utf8)
        let dest = dir.appendingPathComponent(csv.lastPathComponent)
        return (try? fm.moveItem(at: csv, to: dest)) != nil ? dest : nil
    }

    /// Remove an item's kept report (its folder) once the operator has agreed to let it go.
    private func discardKept(_ item: Item) {
        guard case .unsaved = item.state, let kept = item.csv, let root = Self.unsavedRoot,
              kept.path.hasPrefix(root.path + "/") else { return }
        try? FileManager.default.removeItem(at: kept.deletingLastPathComponent())
    }

    /// Ask before `count` unsaved reports are thrown away by `action`. True: go ahead.
    private func confirmDiscard(_ count: Int, then action: String) -> Bool {
        guard count > 0 else { return true }
        let alert = NSAlert()
        alert.messageText = "\(count) report\(count == 1 ? " is" : "s are") not saved."
        alert.informativeText = "\(action) will discard \(count == 1 ? "it" : "them"). Use “Save Unsaved CSVs…” to keep them."
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Discard and \(action)")
        return alert.runModal() == .alertSecondButtonReturn
    }



    /// Save every unsaved CSV into a folder the operator picks now.
    func saveUnsaved() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Save Here"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        // Names this run already placed are never reused: those reports stay where they are.
        var used = taken.union(items.compactMap { $0.csv.map { Placement.fold($0.path) } })
        for i in items.indices {
            guard case .unsaved = items[i].state, let kept = items[i].csv else { continue }
            let stem = items[i].url.deletingPathExtension().lastPathComponent
            if let placed = try? Placement.intoFolder(kept, folder: dir, stem: stem, wav: items[i].url, taken: &used) {
                discardKept(items[i])
                items[i].csv = placed
                items[i].state = .done
            }
        }
        if unsavedCount == 0 { status = "All reports saved." }
    }

    private func finished(_ code: Int32, _ reason: Process.TerminationReason) {
        running = false
        process = nil
        for i in items.indices where items[i].state == .running || items[i].state == .waiting {
            items[i].state = cancelled ? .cancelled : .failed("not checked")
        }
        if cancelled {
            status = "Cancelled."
        } else if reason == .uncaughtSignal {
            status = "The engine stopped unexpectedly."
            errorText = "The engine was stopped by signal \(code)."
        } else if code != 0 {
            status = "The engine stopped with an error."
            let tail = stderrFile.flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? ""
            let lines = tail.split(separator: "\n").suffix(8).joined(separator: "\n")
            errorText = lines.isEmpty ? "Exit code \(code)." : lines
        } else {
            progress = 1
        }
        if unsavedCount > 0 {
            status += " \(unsavedCount) report\(unsavedCount == 1 ? "" : "s") could not be saved — use “Save Unsaved CSVs…”."
        }
        if let workDir { try? FileManager.default.removeItem(at: workDir) }
        workDir = nil
        endAutorunIfNeeded()
    }

    private func endAutorunIfNeeded() {
        guard quitWhenDone else { return }
        let lines = [status, errorText ?? "", runNotes.joined(separator: "; ")]
            + items.map { "\($0.url.lastPathComponent)\t\($0.state)\t\($0.csv?.path ?? "-")\t\($0.summary)\t\($0.notes.joined(separator: "; "))" }
        try? lines.joined(separator: "\n").write(to: FileManager.default.temporaryDirectory
            .appendingPathComponent("autorun_result.txt"), atomically: true, encoding: .utf8)
        NSApp.terminate(nil)
    }

    private static let runPrefix = "fpab-run-"
    private static let ownerFile = "owner.pid"

    /// Working folders left by a crash or a forced quit (their CSVs name client files): removed at
    /// launch — only this app's own, and never one a running copy of the app still uses.
    private static func sweepOldRuns() {
        let fm = FileManager.default
        for url in (try? fm.contentsOfDirectory(at: fm.temporaryDirectory, includingPropertiesForKeys: nil)) ?? []
        where url.lastPathComponent.hasPrefix(runPrefix) || url.lastPathComponent.hasPrefix("run-") {
            if let text = try? String(contentsOf: url.appendingPathComponent(ownerFile), encoding: .utf8),
               let pid = Int32(text), pid != getpid(), kill(pid, 0) == 0 {
                continue                                        // another copy of the app is running it
            }
            if url.lastPathComponent.hasPrefix("run-"),
               !fm.fileExists(atPath: url.appendingPathComponent("engine-stderr.txt").path) {
                continue                                        // not one of ours (an older name)
            }
            try? fm.removeItem(at: url)
        }
    }

    /// Reports an earlier session could not place (it quit or stopped first), not a running copy's.
    private static func leftoverReports() -> (entries: [URL], csvs: [URL]) {
        let fm = FileManager.default
        guard let root = unsavedRoot else { return ([], []) }
        var entries: [URL] = [], csvs: [URL] = []
        for entry in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] {
            var isDir: ObjCBool = false
            fm.fileExists(atPath: entry.path, isDirectory: &isDir)
            if isDir.boolValue {
                if let text = try? String(contentsOf: entry.appendingPathComponent(ownerFile), encoding: .utf8),
                   let pid = Int32(text), pid != getpid(), kill(pid, 0) == 0 {
                    continue                                    // a running copy's
                }
                let inside = (try? fm.contentsOfDirectory(at: entry, includingPropertiesForKeys: nil)) ?? []
                csvs += inside.filter { $0.pathExtension.lowercased() == "csv" }
            } else if entry.pathExtension.lowercased() == "csv" {
                csvs.append(entry)                              // kept by an earlier version, one level up
            }
            entries.append(entry)
        }
        return (entries, csvs)
    }

    /// At launch: offer reports an earlier session could not place — they are never thrown away unasked.
    func offerLeftoverReports() {
        let (entries, csvs) = Self.leftoverReports()
        guard !entries.isEmpty else { return }
        let fm = FileManager.default
        guard !csvs.isEmpty else { entries.forEach { try? fm.removeItem(at: $0) }; return }
        let alert = NSAlert()
        alert.messageText = "\(csvs.count) report\(csvs.count == 1 ? " was" : "s were") never saved."
        alert.informativeText = "The app closed before \(csvs.count == 1 ? "it" : "they") could be placed. "
            + "Save \(csvs.count == 1 ? "it" : "them") to a folder now, or discard \(csvs.count == 1 ? "it" : "them")."
        alert.addButton(withTitle: "Save…")
        alert.addButton(withTitle: "Later")
        alert.addButton(withTitle: "Discard")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.canCreateDirectories = true
            panel.prompt = "Save Here"
            guard panel.runModal() == .OK, let dir = panel.url else { return }
            var all = true
            for csv in csvs {
                let stem = csv.deletingPathExtension().lastPathComponent
                let dest = (1...).lazy.map { k in dir.appendingPathComponent(k == 1 ? "\(stem).csv" : "\(stem) (\(k)).csv") }
                    .first { !fm.fileExists(atPath: $0.path) && (try? fm.destinationOfSymbolicLink(atPath: $0.path)) == nil }!
                if (try? fm.copyItem(at: csv, to: dest)) == nil { all = false }
            }
            if all { entries.forEach { try? fm.removeItem(at: $0) } }
            status = all ? "Saved \(csvs.count) earlier report\(csvs.count == 1 ? "" : "s")." : "Some earlier reports could not be saved."
        case .alertThirdButtonReturn:
            entries.forEach { try? fm.removeItem(at: $0) }
        default:
            break                                               // Later: offered again next time
        }
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
