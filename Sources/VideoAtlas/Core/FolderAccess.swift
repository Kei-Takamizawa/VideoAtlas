// macOSのフォルダ許可ブックマークと動画ファイル列挙を扱います。
import Foundation
// セキュリティスコープを使うフォルダアクセス機能です。
enum FolderAccess {
    // 選択したフォルダURLを永続ブックマークデータへ変換します。
    static func createBookmark(url: URL) throws -> Data {
        // macOSではセキュリティスコープ付きブックマークを作成します。
        #if os(macOS)
        // 後からサンドボックス外のフォルダを再度開ける設定を指定します。
        return try url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
        // macOS以外では通常のブックマークを作成します。
        #else
        // 他のAppleプラットフォームで利用できる標準APIです。
        return try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
        // macOS用の条件付き処理をここで閉じます。
        #endif
    // ここで直前の処理の範囲を閉じます。
    }
    // ブックマークからフォルダURLを解決し、セキュリティスコープを開始します。
    static func resolve(_ source: SourceFolder) throws -> URL {
        // ブックマークが古くなった場合に更新できるかを受け取ります。
        var isStale = false
        // macOSではセキュリティスコープを有効にしてURLを復元します。
        #if os(macOS)
        // 保存時に作った権限付きブックマークを解決します。
        let url = try URL(resolvingBookmarkData: source.bookmark, options: [.withSecurityScope], relativeTo: nil, bookmarkDataIsStale: &isStale)
        // macOS用の条件付き処理をここで閉じます。
        #else
        // 他のAppleプラットフォームでは通常のブックマークを解決します。
        let url = try URL(resolvingBookmarkData: source.bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &isStale)
        // 他プラットフォーム用の条件付き処理をここで閉じます。
        #endif
        // macOSでは読み取りアクセスを開始し、失敗を明示します。
        #if os(macOS)
        // 呼び出し側は利用終了時にstopAccessingSecurityScopedResourceを呼ぶ必要があります。
        guard url.startAccessingSecurityScopedResource() else { throw CocoaError(.fileReadNoPermission) }
        // macOS用の条件付き処理をここで閉じます。
        #endif
        // 解決済みフォルダを呼び出し側へ返します。
        return url
    // ここで直前の処理の範囲を閉じます。
    }
    // フォルダ以下からmp4、mov、m4vを再帰的に読み取り列挙します。
    static func enumerateVideos(in folder: URL) throws -> [URL] {
        // ファイルシステム列挙に使う標準マネージャーです。
        let fileManager = FileManager.default
        // 隠しファイルは対象外にし、フォルダ配下は再帰的に確認します。
        guard let enumerator = fileManager.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { throw CocoaError(.fileReadNoPermission) }
        // 見つかった動画URLを蓄積します。
        var videos: [URL] = []
        // 列挙対象を順番に読み取ります。
        for case let url as URL in enumerator {
            // シンボリックリンクを追跡しないため、リンクそのものを除きます。
            guard (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { continue }
            // 大文字小文字を区別せず動画拡張子を確認します。
            let extensionName = url.pathExtension.lowercased()
            // 指定された3種類だけを対象にします。
            guard ["mp4", "mov", "m4v"].contains(extensionName) else { continue }
            // 実体が通常ファイルのときだけ結果に追加します。
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            // 条件に合った動画URLを追加します。
            videos.append(url)
        // ここで直前の処理の範囲を閉じます。
        }
        // 結果を安定したパス順で返します。
        return videos.sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
