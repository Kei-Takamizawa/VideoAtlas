// このファイルはライブラリ内で共有するデータ型を定義します。
import Foundation
// フォルダを識別し、後で再度開くためのブックマークも保存します。
struct SourceFolder: Identifiable, Codable {
    // フォルダを一意に識別するIDです。
    var id: UUID
    // 選択されたフォルダの表示用パスです。
    var path: String
    // macOSの権限を再取得するためのブックマークデータです。
    var bookmark: Data
// ここで直前の処理の範囲を閉じます。
}
// 動画解析の現在の状態を表す文字列値です。
enum VideoState: String, Codable {
    // まだ解析していない状態です。
    case pending
    // 解析処理を実行中の状態です。
    case analyzing
    // 解析が完了した状態です。
    case ready
    // 元ファイルが見つからない状態です。
    case missing
    // 解析に失敗した状態です。
    case failed
// ここで直前の処理の範囲を閉じます。
}
// 動画ファイルと解析結果の概要を保存します。
struct VideoRecord: Identifiable, Codable {
    // 動画レコードを一意に識別するIDです。
    var id: UUID
    // 動画を含む登録元フォルダのIDです。
    var sourceID: UUID
    // 動画ファイルのパスです。
    var path: String
    // 動画ファイル名です。
    var name: String
    // 動画の長さを秒で保存します。
    var duration: Double
    // 動画ファイルのサイズをバイトで保存します。
    var fileSize: Int64
    // ファイルの最終更新日時です。
    var modificationTime: Date
    // 解析状態です。
    var state: VideoState
    // 最後に解析した位置を秒で保存します。
    var lastAnalyzedSecond: Double
    // 解析時に使うサンプル間隔を秒で保存します。
    var sampleInterval: Double
    // 失敗などの説明があれば保存します。
    var errorMessage: String?
    // 生成済みポスター画像のパスがあれば保存します。
    var posterPath: String?
// ここで直前の処理の範囲を閉じます。
}
// 1枚の顔検出結果と人物割当情報を保存します。
struct FaceRecord: Identifiable, Codable {
    // 顔レコードを一意に識別するIDです。
    var id: UUID
    // この顔が検出された動画のIDです。
    var videoID: UUID
    // 動画内で検出された時刻を秒で保存します。
    var second: Double
    // 動画フレーム内で顔が占める相対位置と大きさです。
    var boundingBox: CGRect? = nil
    // 手動または自動で割り当てられた人物IDです。
    var personID: UUID?
    // 顔サムネイル画像のパスです。
    var thumbnailPath: String
    // 顔の特徴量があれば浮動小数点配列で保存します。
    var embedding: [Float]?
    // 人物割当を人が手動で変更したかを表します。
    var manualAssignment: Bool
    // この顔を一覧や集計から除外するかを表します。
    var excluded: Bool
// ここで直前の処理の範囲を閉じます。
}
// 顔をまとめる人物の表示情報を保存します。
struct PersonRecord: Identifiable, Codable {
    // 人物を一意に識別するIDです。
    var id: UUID
    // 人物名です。
    var name: String
    // 人物の代表サムネイルがあればそのパスです。
    var thumbnailPath: String?
// ここで直前の処理の範囲を閉じます。
}
// データベースから読み込んだ全レコードをまとめます。
struct LibrarySnapshot {
    // 登録済みフォルダ一覧です。
    var sources: [SourceFolder]
    // 登録済み動画一覧です。
    var videos: [VideoRecord]
    // 検出済み顔一覧です。
    var faces: [FaceRecord]
    // 登録済み人物一覧です。
    var people: [PersonRecord]
    // すべて空のライブラリを作るための初期化子です。
    init(sources: [SourceFolder] = [], videos: [VideoRecord] = [], faces: [FaceRecord] = [], people: [PersonRecord] = []) {
        // フォルダ一覧を保存します。
        self.sources = sources
        // 動画一覧を保存します。
        self.videos = videos
        // 顔一覧を保存します。
        self.faces = faces
        // 人物一覧を保存します。
        self.people = people
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
