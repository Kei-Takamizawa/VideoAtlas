// 動画パス、時計、JSON を扱う標準機能を読み込みます。
import Foundation
// コマンド用の検証で付属リソースを見つける方法を定義します。
extension Bundle {
    // 二番目の引数で指定したアプリ付属リソースを読み込みます。
    static let module = Bundle(path: CommandLine.arguments[2])!
// リソース参照の定義を閉じます。
}
// 任意の時刻で測定を止めるスレッド安全な状態です。
final class BenchmarkStop: @unchecked Sendable {
    // 解析処理と通知処理の間の値を保護します。
    private let lock = NSLock()
    // 停止時刻に達したか保持します。
    private var stopped = false
    // 停止要求を書き込みます。
    func update(_ value: Bool) {
        // 排他的にアクセスします。
        lock.lock()
        // 停止状態を更新します。
        stopped = value
        // ロックを解除します。
        lock.unlock()
    // 更新処理を閉じます。
    }
    // 停止状態を読み込みます。
    func read() -> Bool {
        // 値を読む間だけ保護します。
        lock.lock()
        // 読み終えたら解除します。
        defer { lock.unlock() }
        // 現在値を返します。
        return stopped
    // 読み込み処理を閉じます。
    }
// 停止状態の定義を閉じます。
}
// 同じ動画区間を解析し、処理秒数と検出数だけを出力します。
@main struct Benchmark {
    // 非同期解析の起点です。
    static func main() async throws {
        // 解析用のモデルを一度読み込みます。
        let analyzer = try VideoAnalyzer()
        // 指定された動画の場所を取得します。
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        // 三番目の引数は測定する動画内の秒数です。
        let end = Double(CommandLine.arguments[3])!
        // 終了通知を共有する状態を作ります。
        let stop = BenchmarkStop()
        // モデル読み込み後から実際の解析時間を測ります。
        let started = Date()
        // 二秒間隔で同じ区間を調べ、指定位置で止めます。
        let result = try await analyzer.analyze(url: url, interval: 2, startingAt: 0, endingAt: end, progress: { second, _ in stop.update(second >= end) }, isCancelled: { stop.read() })
        // 顔画像や特徴量は出力せず、集計値だけを作ります。
        let metrics: [String: Any] = ["file": url.lastPathComponent, "wallSeconds": Date().timeIntervalSince(started), "completedSecond": result.completedSecond, "faceCount": result.faces.count, "embeddings": result.faces.filter { $0.embedding != nil }.count]
        // 後から比較できる JSON へ変換します。
        let data = try JSONSerialization.data(withJSONObject: metrics, options: [.sortedKeys])
        // 一行の測定結果を標準出力へ書き込みます。
        print(String(decoding: data, as: UTF8.self))
    // 測定の起点を閉じます。
    }
// コマンドの定義を閉じます。
}
