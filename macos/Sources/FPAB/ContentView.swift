import SwiftUI

struct ContentView: View {
    @Bindable var model: RunModel
    @State private var hovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            dropZone
            HStack {
                Button("Add Files…") { model.chooseFiles() }
                Button("Clear") { model.clear() }.disabled(model.items.isEmpty)
                Spacer()
            }
            .disabled(model.running)
            settings
            VStack(alignment: .leading, spacing: 6) {
                ProgressView(value: model.progress)
                Text(model.status).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                if let error = model.errorText {
                    Text(error).font(.caption.monospaced()).foregroundStyle(.red).textSelection(.enabled)
                }
                ForEach(model.runNotes, id: \.self) { note in
                    Label(note, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(.orange)
                }
                if !Engine.isInstalled {
                    Text("The analysis engine is missing from this app. Build it with macos/build_app.sh.")
                        .font(.caption).foregroundStyle(.red)
                }
            }
            HStack {
                if model.unsavedCount > 0 && !model.running {
                    Button("Save Unsaved CSVs…") { model.saveUnsaved() }
                }
                Spacer()
                if model.running {
                    Button("Cancel", role: .cancel) { model.cancel() }
                } else {
                    Button("Go") { model.go() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(!model.canGo)
                }
            }
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: 480)
    }

    private var dropZone: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(hovering ? Color.accentColor : Color.secondary.opacity(0.4),
                              style: StrokeStyle(lineWidth: hovering ? 2 : 1, dash: [6, 4]))
                .background(RoundedRectangle(cornerRadius: 10)
                    .fill(hovering ? Color.accentColor.opacity(0.08) : Color.clear))
            if model.items.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "waveform").font(.system(size: 32)).foregroundStyle(.secondary)
                    Text("Drop WAV files or folders here").foregroundStyle(.secondary)
                }
            } else {
                List {
                    ForEach(model.items) { item in
                        row(item).contextMenu {
                            Button("Remove") { model.remove(item.id) }.disabled(model.running)
                        }
                    }
                }
                .scrollContentBackground(.hidden)
                .padding(4)
            }
        }
        .frame(minHeight: 220)
        .overlay(FileDropTarget(accepts: { !model.running }, onFiles: { model.add($0) },
                                onHover: { hovering = $0 }))
    }

    private func row(_ item: RunModel.Item) -> some View {
        HStack(spacing: 8) {
            icon(item.state)
            Text(item.url.lastPathComponent).lineLimit(1).truncationMode(.middle)
            Spacer()
            if !item.notes.isEmpty {
                Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
                    .help(item.notes.joined(separator: "\n"))
            }
            switch item.state {
            case .unsaved(let why):
                Text("not saved: \(why)").font(.caption).foregroundStyle(.orange).lineLimit(1).help(why)
            case .done:
                Text(item.summary).font(.caption).foregroundStyle(.secondary)
                if let csv = item.csv {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([csv])
                    } label: { Image(systemName: "magnifyingglass") }
                        .buttonStyle(.borderless)
                        .help("Show the CSV in Finder")
                }
            case .failed(let why):
                Text(why).font(.caption).foregroundStyle(.red).lineLimit(1).help(why)
            case .cancelled:
                Text("cancelled").font(.caption).foregroundStyle(.secondary)
            case .running:
                Text("checking…").font(.caption).foregroundStyle(.secondary)
            case .waiting:
                EmptyView()
            }
        }
    }

    @ViewBuilder private func icon(_ state: RunModel.State) -> some View {
        switch state {
        case .waiting: Image(systemName: "circle").foregroundStyle(.secondary)
        case .running: ProgressView().controlSize(.small)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        case .unsaved: Image(systemName: "tray.full").foregroundStyle(.orange)
        case .cancelled: Image(systemName: "minus.circle").foregroundStyle(.secondary)
        }
    }

    private var settings: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Save each CSV", selection: $model.output) {
                    Text("Next to its WAV (same name, .csv)").tag(RunModel.Output.besideWAV)
                    Text("In a folder").tag(RunModel.Output.folder)
                }
                .pickerStyle(.radioGroup)
                if model.output == .folder {
                    HStack {
                        Text(model.folder?.path(percentEncoded: false) ?? "No folder chosen")
                            .font(.callout).foregroundStyle(model.folder == nil ? .red : .secondary)
                            .lineLimit(1).truncationMode(.middle)
                        Button("Choose…") { model.chooseFolder() }
                    }
                    .padding(.leading, 20)
                }
                Toggle("Include pause identification in report", isOn: $model.withPauses)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
        .disabled(model.running)
    }
}
