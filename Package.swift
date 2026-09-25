// swift-tools-version: 6.4
// Swift Package Manager でアプリの構成を記述する機能を読み込みます。
import PackageDescription
// VideoAtlas の実行ファイルとテストの構成を定義します。
let package = Package(
    // パッケージの名前を指定します。
    name: "VideoAtlas",
    // Liquid Glass を使用できる macOS 26 以降を対象にします。
    platforms: [.macOS(.v26)],
    // Swift の実行ファイルを外部へ公開します。
    products: [.executable(name: "VideoAtlas", targets: ["VideoAtlas"])],
    // ソースコード、付属モデル、テストをビルド対象にします。
    targets: [
        // SwiftUI アプリ本体を実行ファイルとしてビルドします。
        .executableTarget(
            // アプリ本体のターゲット名です。
            name: "VideoAtlas",
            // モデルと実行ライブラリをアプリのリソースに含めます。
            resources: [.copy("Resources/Models"), .copy("Resources/OpenVINO"), .copy("Resources/Licenses")],
            // SQLite の C ライブラリをデータベース処理に結び付けます。
            linkerSettings: [.linkedLibrary("sqlite3")]
        // ここで設定した引数のまとまりを閉じます。
        ),
        // 解析とデータ保存の重要な振る舞いを検証します。
        .testTarget(name: "VideoAtlasTests", dependencies: ["VideoAtlas"])
    // ここで項目の一覧を閉じます。
    ]
// ここで設定した引数のまとまりを閉じます。
)
