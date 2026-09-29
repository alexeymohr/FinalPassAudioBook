import AppKit
import SwiftUI

/// Finder file drops via AppKit's NSDraggingDestination: it fires the instant the mouse is
/// released (SwiftUI's drop modifiers add a settle delay). It passes clicks through (hitTest nil),
/// so the list underneath stays usable.
final class FileDropView: NSView {
    var onFiles: (([URL]) -> Void)?
    var onHover: ((Bool) -> Void)?
    var accepts: (() -> Bool)?                  // false during a run: the drop is refused, not lost

    override init(frame: NSRect) {
        super.init(frame: frame)
        registerForDraggedTypes([.fileURL])
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    private func urls(_ info: NSDraggingInfo) -> [URL] {
        info.draggingPasteboard.readObjects(forClasses: [NSURL.self],
                                            options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
    }

    private func acceptable(_ info: NSDraggingInfo) -> Bool { (accepts?() ?? true) && !urls(info).isEmpty }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        let ok = acceptable(sender)
        onHover?(ok)
        return ok ? .copy : []
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        acceptable(sender) ? .copy : []
    }

    override func draggingExited(_ sender: NSDraggingInfo?) { onHover?(false) }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        onHover?(false)
        guard acceptable(sender) else { return false }
        let found = urls(sender)
        onFiles?(found)
        return true
    }
}

struct FileDropTarget: NSViewRepresentable {
    let accepts: () -> Bool
    let onFiles: ([URL]) -> Void
    let onHover: (Bool) -> Void

    func makeNSView(context: Context) -> FileDropView {
        let view = FileDropView()
        updateNSView(view, context: context)
        return view
    }

    func updateNSView(_ view: FileDropView, context: Context) {
        view.accepts = accepts
        view.onFiles = onFiles
        view.onHover = onHover
    }
}
