// swift-tools-version: 5.9
// Copyright (c) 2026 Cleo. Licensed under Apache-2.0.

import PackageDescription

let package = Package(
    name: "TribeBrainView",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "TribeBrainView", targets: ["TribeBrainView"]),
        .library(name: "TribeBrainViewUI", targets: ["TribeBrainViewUI"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "TribeBrainView",
            dependencies: [],
            path: "Sources/TribeBrainView"
        ),
        .target(
            name: "TribeBrainViewUI",
            dependencies: ["TribeBrainView"],
            path: "Sources/TribeBrainViewUI"
        ),
        .testTarget(
            name: "TribeBrainViewTests",
            dependencies: ["TribeBrainView"],
            path: "Tests",
            exclude: ["TribeBrainViewUITests"],
            sources: ["TribeBrainViewTests"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
        .testTarget(
            name: "TribeBrainViewUITests",
            dependencies: ["TribeBrainViewUI"],
            path: "Tests/TribeBrainViewUITests"
        ),
    ]
)
