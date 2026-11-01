// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "R2App",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "R2App", path: "Sources/R2App")
    ]
)
