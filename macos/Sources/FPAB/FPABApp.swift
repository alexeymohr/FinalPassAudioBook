import SwiftUI

@main
struct FPABApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("FinalPass AudioBook", id: "main") {
            ContentView(model: delegate.model)
        }
        .windowResizability(.contentMinSize)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = RunModel()

    /// WAVs dropped on the Dock icon or opened with the app. Ignored (with a beep) during a run.
    func application(_ application: NSApplication, open urls: [URL]) {
        guard !model.running else { NSSound.beep(); return }
        model.add(urls)
        #if FPAB_TEST_HOOKS
        // Test builds only: `open --env FPAB_AUTORUN=1 -a "FinalPass AudioBook" a.wav b.wav` runs and quits.
        if ProcessInfo.processInfo.environment["FPAB_AUTORUN"] == "1" { model.autorun() }
        #endif
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationDidFinishLaunching(_ notification: Notification) {
        #if FPAB_TEST_HOOKS
        if ProcessInfo.processInfo.environment["FPAB_AUTORUN"] == "1" { return }     // never a modal in a scripted run
        #endif
        model.offerLeftoverReports()
    }

    /// Unsaved reports are not lost on quitting: they are offered again at the next launch.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard model.running else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "A check is still running."
        alert.informativeText = "Quit now and stop it? Files already finished keep their CSVs; reports not yet "
            + "saved are offered again when the app next opens."
        alert.addButton(withTitle: "Keep Running")
        alert.addButton(withTitle: "Quit")
        return alert.runModal() == .alertSecondButtonReturn ? .terminateNow : .terminateCancel
    }

    func applicationWillTerminate(_ notification: Notification) { model.stopForQuit() }
}
