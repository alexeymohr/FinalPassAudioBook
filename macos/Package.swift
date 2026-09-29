// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FPAB",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "FPAB", path: "Sources/FPAB")
    ]
)
