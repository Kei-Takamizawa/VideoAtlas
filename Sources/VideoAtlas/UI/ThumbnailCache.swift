// サムネイルを小さく読み込み、同じ画像の再読み込みを避けるための機能を読み込みます。
import AppKit
// 非同期読み込み済み画像を画面部品として表示する SwiftUI を読み込みます。
import SwiftUI
// ファイルから必要な大きさの画像だけを作る ImageIO を読み込みます。
import ImageIO
// キャッシュと画像デコードを複数スレッドから安全に行う型を定義します。
final class ThumbnailCache: @unchecked Sendable {
    // アプリ内でサムネイルキャッシュを一つだけ共有します。
    static let shared = ThumbnailCache()
    // 同じ画像の参照を素早く返すメモリキャッシュを保持します。
    private let images = NSCache<NSString, NSImage>()
    // キャッシュの登録数と概算メモリ量を抑える設定を行います。
    private init() {
        // 画面に必要な画像だけを残すため保存数を最大 240 件にします。
        images.countLimit = 240
        // サムネイルの合計メモリ使用量をおよそ 96 MiB に制限します。
        images.totalCostLimit = 96 * 1024 * 1024
    // キャッシュ設定の定義を閉じます。
    }
    // サムネイルをバックグラウンドで読み込んで返します。
    func image(at path: String, maximumPixelSize: Int) async -> NSImage? {
        // パスと最大辺の長さを組み合わせてサイズ違いを区別するキーを作ります。
        let key = "\(path)#\(maximumPixelSize)" as NSString
        // すでに同じ大きさの画像があればディスクに触れず返します。
        if let cached = images.object(forKey: key) { return cached }
        // ファイル読み込みと画像縮小を画面描画スレッドから切り離します。
        let loaded = await Task.detached(priority: .utility) { () -> NSImage? in
            // 画像ファイルをデコードせず読み取れるソースを作ります。
            guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
            // 元画像全体ではなく指定サイズ以下の縮小画像を生成します。
            let options = [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize, kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary
            // 読み込みと回転補正を行い、失敗時は画像なしとして返します。
            guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options) else { return nil }
            // 元ピクセル寸法を保持した NSImage にして SwiftUI の表示寸法を安定させます。
            return NSImage(cgImage: thumbnail, size: NSSize(width: thumbnail.width, height: thumbnail.height))
        // バックグラウンド処理の定義を閉じます。
        }.value
        // 読み込みに成功した画像だけをキャッシュに登録します。
        if let loaded {
            // 縮小画像を一度だけピクセル形式へ取り出して概算コストを求めます。
            let bitmap = loaded.cgImage(forProposedRect: nil, context: nil, hints: nil)
            // ピクセル数に基づく概算コストを付けてキャッシュへ保存します。
            images.setObject(loaded, forKey: key, cost: (bitmap?.bytesPerRow ?? 0) * (bitmap?.height ?? 0))
        // 成功画像のキャッシュ条件を閉じます。
        }
        // 画像または読み込み失敗を呼び出し側へ返します。
        return loaded
    // 非同期読み込み関数の定義を閉じます。
    }
// キャッシュ型の定義を閉じます。
}
// 非同期で読み込んだ画像を SwiftUI で表示する小さな部品です。
struct CachedThumbnail: View {
    // 画像ファイルの場所を受け取り、未設定ならプレースホルダーを表示します。
    let path: String?
    // 正方形の表示サイズを受け取ります。
    let size: CGFloat
    // 画像がない場合に表示する SF Symbols 名を受け取ります。
    let placeholder: String
    // 読み込み済み画像を画面状態として保持します。
    @State private var image: NSImage?
    // 画像またはプレースホルダーを組み立てます。
    var body: some View {
        // 画像が読み込まれるまで同じ大きさの表示領域を確保します。
        Group {
            // 画像が読み込まれていれば縮小済み画像を表示します。
            if let image {
                // 呼び出し側の形状加工に合わせて指定サイズへ収めます。
                Image(nsImage: image).resizable().scaledToFill().frame(width: size, height: size)
            // 読み込み前または失敗時は人物アイコンを表示します。
            } else {
                // 読み込み中もレイアウトが変わらないよう同じ寸法を使います。
                Image(systemName: placeholder).font(.system(size: size * 0.62)).foregroundStyle(.secondary).frame(width: size, height: size).background(.quaternary)
            // 画像有無による表示分岐を閉じます。
            }
        // 画像とプレースホルダーをまとめる Group を閉じます。
        }
        // ファイルの読み込みを描画処理とは別の非同期タスクとして開始します。
        .task(id: path) {
            // ファイル場所が変わったとき古い画像を一時表示しないよう消します。
            image = nil
            // パスがある場合だけ適切な大きさで縮小して読み込みます。
            if let path {
                // 表示サイズの 2 倍の画素数で読み込み、Retina 表示にも対応します。
                let loaded = await ThumbnailCache.shared.image(at: path, maximumPixelSize: Int(size * 2))
                // 表示対象が切り替わった後の古い非同期結果は画面へ反映しません。
                if !Task.isCancelled { image = loaded }
            // パスがある場合の読み込み条件を閉じます。
            }
        // パスに結び付いた非同期処理を閉じます。
        }
    // サムネイル表示部品の本文を閉じます。
    }
// SwiftUI サムネイル部品の定義を閉じます。
}
