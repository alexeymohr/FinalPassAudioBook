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
        for url in urls {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                let inside = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isRegularFileKey],
                                                          options: [.skipsHiddenFiles])) ?? []
                let wavs = inside.filter { Self.isAudioFile($0) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
                if wavs.isEmpty { emptyFolders.append(url.lastPathComponent) }
                found += wavs
            } else if Self.isAudioFile(url) {
                found.append(url)
            }
        }
        let known = Set(items.map(\.url.standardizedFileURL))
        items += found.filter { !known.contains($0.standardizedFileURL) }.map { Item(url: $0) }
        if !emptyFolders.isEmpty {
            status = "No WAV files found in “\(emptyFolders.joined(separator: "”, “"))” (sub-folders are not searched)."
        } else if !items.isEmpty {
            status = "\(items.count) file\(items.count == 1 ? "" : "s") ready."
        }
    }

    /// A regular file (not a folder, even one named "x.wav") with a WAV/BWF extension.
    private static func isAudioFile(_ url: URL) -> Bool {
        audioExtensions.contains(url.pathExtension.lowercased())
            && (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
    }

    func remove(_ id: UUID) { if !running { items.removeAll { $0.id == id } } }

    func clear() {
        guard !running else { return }
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

    /// A saved security-scoped bookmark, refreshed when macOS reports it stale.
    private static func resolveBookmark(_ data: Data?, key: String?) -> URL? {
        guard let data else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, relativeTo: nil,
                                 bookmarkDataIsStale: &stale) else { return nil }
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
        guard canGo else { return }
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
        let work = fm.temporaryDirectory.appendingPathComponent("run-\(UUID().uuidString)", isDirectory: true)
        let errFile = work.appendingPathComponent("engine-stderr.txt")
        do {
            try fm.createDirectory(at: work, withIntermediateDirectories: true)
            fm.createFile(atPath: errFile.path, contents: nil)
        } catch {
            errorText = "Could not create a working folder: \(error.localizedDescription)"
            return
        }
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
                items[i].csv = keepUnsaved(csv)
                items[i].state = .unsaved(error.localizedDescription)
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
            return try Placement.intoFolder(csv, folder: folder, stem: stem, taken: &taken)
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
            return try Placement.intoFolder(csv, folder: alt, stem: stem, taken: &taken)
        }
    }

    /// A CSV that could not be put in place is kept in the app's own storage, never thrown away.
    private func keepUnsaved(_ csv: URL) -> URL? {
        let fm = FileManager.default
        guard let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return nil }
        let dir = base.appendingPathComponent("Unsaved Reports", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let dest = dir.appendingPathComponent("\(UUID().uuidString.prefix(8)) \(csv.lastPathComponent)")
        return (try? fm.moveItem(at: csv, to: dest)) != nil ? dest : nil
    }

    /// Save every unsaved CSV into a folder the operator picks now.
    func saveUnsaved() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Save Here"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        var used = Set<String>()
        for i in items.indices {
            guard case .unsaved = items[i].state, let kept = items[i].csv else { continue }
            let stem = items[i].url.deletingPathExtension().lastPathComponent
            if let placed = try? Placement.intoFolder(kept, folder: dir, stem: stem, taken: &used) {
                try? FileManager.default.removeItem(at: kept)
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

    /// Working folders left by a crash or a forced quit (their CSVs name client files): removed at launch.
    private static func sweepOldRuns() {
        let fm = FileManager.default
        for url in (try? fm.contentsOfDirectory(at: fm.temporaryDirectory, includingPropertiesForKeys: nil)) ?? []
        where url.lastPathComponent.hasPrefix("run-") {
            try? fm.removeItem(at: url)
        }
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
