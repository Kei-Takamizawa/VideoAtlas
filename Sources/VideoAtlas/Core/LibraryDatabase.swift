// SQLiteを使ってアプリのメタデータを永続化します。
import Foundation
// SQLiteのC APIをSwiftから利用します。
import SQLite3
// データベースの初期化や読み書きで発生するエラーをまとめます。
enum LibraryDatabaseError: Error {
    // SQLiteが返したエラー内容を保持します。
    case sqlite(String)
    // JSONへの変換に失敗した場合に使います。
    case encoding(String)
// ここで直前の処理の範囲を閉じます。
}
// 動画ライブラリのSQLiteデータベースを操作します。
final class LibraryDatabase {
    // SQLite接続ハンドルを保持します。
    private var handle: OpaquePointer?
    // CodableのJSON変換で日付を数値として保存します。
    private let encoder = JSONEncoder()
    // 保存されたJSONを読み戻す変換器です。
    private let decoder = JSONDecoder()
    // 指定されたURLでデータベースを開き、必要なテーブルを準備します。
    init(databaseURL: URL) throws {
        // SQLiteファイルを読み書きモードで開きます。
        let openResult = sqlite3_open(databaseURL.path, &handle)
        // ファイルを開けなかった場合は原因を返します。
        guard openResult == SQLITE_OK else { throw makeError() }
        // 他の接続が書き込み中でも最大5秒待つよう設定します。
        sqlite3_busy_timeout(handle, 5_000)
        // 外部キー制約を有効にし、関連データの整合性を守ります。
        try execute("PRAGMA foreign_keys = ON")
        // 画面の読み取りと解析結果の書き込みが互いを長く待たないようにします。
        try execute("PRAGMA journal_mode = WAL")
        // 登録フォルダ用テーブルを作成します。
        try execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
        // 動画用テーブルを作り、フォルダ削除時に動画も削除します。
        try execute("CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY NOT NULL, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE, payload TEXT NOT NULL)")
        // 人物用テーブルを作成します。
        try execute("CREATE TABLE IF NOT EXISTS people (id TEXT PRIMARY KEY NOT NULL, payload TEXT NOT NULL)")
        // 顔用テーブルを作り、動画削除時に顔も削除します。
        try execute("CREATE TABLE IF NOT EXISTS faces (id TEXT PRIMARY KEY NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE, person_id TEXT REFERENCES people(id) ON DELETE SET NULL, payload TEXT NOT NULL)")
        // 自動統合を既存ライブラリへ一度だけ適用した版を保存します。
        try execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)")
    // ここで直前の処理の範囲を閉じます。
    }
    // 接続を閉じてSQLiteリソースを解放します。
    deinit {
        // 開いている接続があれば閉じます。
        sqlite3_close(handle)
    // ここで直前の処理の範囲を閉じます。
    }
    // 全テーブルからレコードを読み込んでライブラリ状態を返します。
    func load() throws -> LibrarySnapshot {
        // 各テーブルのJSONを型ごとに復元してまとめます。
        return LibrarySnapshot(sources: try readAll("sources", as: SourceFolder.self), videos: try readAll("videos", as: VideoRecord.self), faces: try readAll("faces", as: FaceRecord.self), people: try readAll("people", as: PersonRecord.self))
    // ここで直前の処理の範囲を閉じます。
    }
    // 現行版の自動統合を既存ライブラリへ適用済みか調べます。
    func hasCurrentGroupReconciliation() throws -> Bool {
        // 版情報を読み取る SQLite 文を用意します。
        var statement: OpaquePointer?
        // 固定されたメタデータのキーだけを読みます。
        guard sqlite3_prepare_v2(handle, "SELECT value FROM metadata WHERE key = 'group_reconciliation_version'", -1, &statement, nil) == SQLITE_OK else { throw makeError() }
        // 読み取りが終わったら文を解放します。
        defer { sqlite3_finalize(statement) }
        // 行が存在しなければ旧ライブラリとして未適用にします。
        guard sqlite3_step(statement) == SQLITE_ROW, let value = sqlite3_column_text(statement, 0) else { return false }
        // 現在の照合規則の版だけを適用済みとみなします。
        return String(cString: value) == "2"
    // ここで直前の処理の範囲を閉じます。
    }
    // 現在の自動統合を実行済みと記録します。
    func markCurrentGroupReconciliation() throws {
        // 同じアプリを再起動しても重い照合を繰り返さないよう保存します。
        try execute("INSERT INTO metadata(key, value) VALUES ('group_reconciliation_version', '2') ON CONFLICT(key) DO UPDATE SET value = excluded.value")
    // ここで直前の処理の範囲を閉じます。
    }
    // フォルダ情報を追加または更新します。
    func saveSource(_ source: SourceFolder) throws {
        // フォルダIDとJSON本文を保存します。
        try upsert(table: "sources", id: source.id, value: source)
    // ここで直前の処理の範囲を閉じます。
    }
    // 動画情報を追加または更新します。
    func saveVideo(_ video: VideoRecord) throws {
        // フォルダとの関連を含めて動画JSONを保存します。
        try upsert(table: "videos", id: video.id, parentColumn: "source_id", parentID: video.sourceID, value: video)
    // ここで直前の処理の範囲を閉じます。
    }
    // 内容が変わった動画の情報と旧顔の削除を一つの処理で確定します。
    func invalidateVideoRevision(_ video: VideoRecord) throws -> Set<UUID> {
        // 動画更新と旧顔削除を同時に確定するためトランザクションを開始します。
        try execute("BEGIN IMMEDIATE")
        // 途中で失敗した場合にロールバックするための状態を保持します。
        var committed = false
        // 確定していなければ変更を元に戻します。
        defer { if !committed { try? execute("ROLLBACK") } }
        // 更新前にこの動画の顔が参照していた人物を調べます。
        let candidates = try personIDsForVideo(video.id)
        // 新しいファイル属性と未解析状態を保存します。
        try saveVideo(video)
        // 以前の内容から検出された顔をすべて削除します。
        try execute("DELETE FROM faces WHERE video_id = '\(video.id.uuidString)'")
        // この動画の顔を消した後に参照がなくなる人物を記録します。
        var removedPeople = Set<UUID>()
        // 旧顔が参照していた人物だけを削除対象として調べます。
        for personID in candidates {
            // 他の動画の顔が参照している人物は残します。
            try execute("DELETE FROM people WHERE id = '\(personID.uuidString)' AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = '\(personID.uuidString)')")
            // 削除された場合だけ画面側にも通知します。
            if sqlite3_changes(handle) > 0 { removedPeople.insert(personID) }
        // ここで直前の処理の範囲を閉じます。
        }
        // 動画・顔・人物の変更をまとめて確定します。
        try execute("COMMIT")
        // 確定後のロールバックを抑止します。
        committed = true
        // 実際に削除した人物の ID を返します。
        return removedPeople
    // ここで直前の処理の範囲を閉じます。
    }
    // 人物情報を追加または更新します。
    func savePerson(_ person: PersonRecord) throws {
        // 人物IDとJSON本文を保存します。
        try upsert(table: "people", id: person.id, value: person)
    // ここで直前の処理の範囲を閉じます。
    }
    // 顔情報を追加または更新します。
    func saveFace(_ face: FaceRecord) throws {
        // 動画と人物への外部キーを含めて保存します。
        try upsert(table: "faces", id: face.id, parentColumn: "video_id", parentID: face.videoID, nullableColumn: "person_id", nullableID: face.personID, value: face)
    // ここで直前の処理の範囲を閉じます。
    }
    // 指定動画の顔一覧を1つのトランザクションで置き換えます。
    func replaceFaces(videoID: UUID, faces: [FaceRecord]) throws {
        // 置き換え前に、渡された各顔が対象動画のものか確認します。
        guard faces.allSatisfy({ $0.videoID == videoID }) else { throw LibraryDatabaseError.encoding("顔レコードのvideoIDが対象動画と一致しません") }
        // 削除と追加の途中状態を外部から見せないようトランザクションを開始します。
        try execute("BEGIN IMMEDIATE")
        // 途中で失敗したら変更を取り消すためのフラグです。
        var committed = false
        // 関数を抜けるとき、完了していなければ自動でロールバックします。
        defer { if !committed { try? execute("ROLLBACK") } }
        // 対象動画の既存顔を削除します。
        try execute("DELETE FROM faces WHERE video_id = '\(videoID.uuidString)'" )
        // 新しい顔を順番に保存します。
        for face in faces { try saveFace(face) }
        // すべて成功した場合だけ変更を確定します。
        try execute("COMMIT")
        // deferのロールバックを抑止します。
        committed = true
    // ここで直前の処理の範囲を閉じます。
    }
    // 解析済みの小さな区間だけを保存し、以前の区間の顔を触らずに残します。
    func applyFaceBatch(video: VideoRecord, newPeople: [PersonRecord], faces: [FaceRecord], removing oldFaceIDs: Set<UUID>) throws {
        // 他の動画の顔が誤って混じることを防ぎます。
        guard faces.allSatisfy({ $0.videoID == video.id }) else { throw LibraryDatabaseError.encoding("顔レコードのvideoIDが対象動画と一致しません") }
        // 動画の再開位置と新しい顔を同時に確定します。
        try execute("BEGIN IMMEDIATE")
        // 保存完了前に失敗したかどうかを保持します。
        var committed = false
        // 失敗した場合は全変更を取り消します。
        defer { if !committed { try? execute("ROLLBACK") } }
        // 新しい顔の参照先となる人物を先に保存します。
        for person in newPeople { try savePerson(person) }
        // 今回再解析した区間の旧顔だけを削除します。
        for identifier in oldFaceIDs { try delete(table: "faces", id: identifier) }
        // 今回見つけた顔を順番に保存します。
        for face in faces { try saveFace(face) }
        // 顔と同じ確定点で動画の再開位置を保存します。
        try saveVideo(video)
        // 一つの区間の結果を確定します。
        try execute("COMMIT")
        // 完了後のロールバックを止めます。
        committed = true
    // ここで直前の処理の範囲を閉じます。
    }
    // 自動で十分に一致した人物グループだけを一つの処理で統合します。
    func applyAutomaticPersonMerges(updatedFaces: [FaceRecord], removing personIDs: Set<UUID>) throws {
        // 手動修正された顔は自動統合の対象にしません。
        guard updatedFaces.allSatisfy({ !$0.manualAssignment }) else { throw LibraryDatabaseError.encoding("手動修正された顔は自動統合できません") }
        // 顔と人物の更新を一度に確定します。
        try execute("BEGIN IMMEDIATE")
        // 保存が最後まで成功したか記録します。
        var committed = false
        // 途中で失敗した場合は以前の人物分類へ戻します。
        defer { if !committed { try? execute("ROLLBACK") } }
        // 各顔の人物 ID を新しい統合先へ保存します。
        for face in updatedFaces { try saveFace(face) }
        // 顔がすべて移動した統合元だけを削除します。
        for personID in personIDs {
            // 参照が残る人物を誤って削除しない SQL を使います。
            try execute("DELETE FROM people WHERE id = '\(personID.uuidString)' AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = '\(personID.uuidString)')")
            // 参照残りや対象の欠落があれば統合全体を取り消します。
            guard sqlite3_changes(handle) == 1 else { throw LibraryDatabaseError.encoding("統合元の人物に顔の参照が残っています") }
        // ここで直前の処理の範囲を閉じます。
        }
        // すべての顔と人物の変更を確定します。
        try execute("COMMIT")
        // 正常終了時はロールバックを止めます。
        committed = true
    // ここで直前の処理の範囲を閉じます。
    }
    // 指定フォルダを削除し、外部キーのCASCADEで動画と顔も削除します。
    func removeSource(id: UUID) throws {
        // フォルダIDで対象行を削除します。
        try delete(table: "sources", id: id)
    // ここで直前の処理の範囲を閉じます。
    }
    // 指定動画を削除し、関連する顔も削除します。
    func removeVideo(id: UUID) throws {
        // 動画IDで対象行を削除します。
        try delete(table: "videos", id: id)
    // ここで直前の処理の範囲を閉じます。
    }
    // 指定人物を削除し、関連する顔の人物IDをNULLにします。
    func removePerson(id: UUID) throws {
        // 人物IDで対象行を削除します。
        try delete(table: "people", id: id)
    // ここで直前の処理の範囲を閉じます。
    }
    // すべての登録情報を空にします。
    func clearAll() throws {
        // 3テーブルの削除を1つの処理として実行します。
        try execute("BEGIN IMMEDIATE")
        // 途中で失敗したときは削除を戻します。
        var committed = false
        // 未確定の処理をロールバックします。
        defer { if !committed { try? execute("ROLLBACK") } }
        // フォルダを削除すると動画と顔は連鎖削除されます。
        try execute("DELETE FROM sources")
        // 孤立した人物を削除します。
        try execute("DELETE FROM people")
        // 空のライブラリでは旧照合の完了印も削除します。
        try execute("DELETE FROM metadata")
        // 削除完了を確定します。
        try execute("COMMIT")
        // deferによるロールバックを止めます。
        committed = true
    // ここで直前の処理の範囲を閉じます。
    }
    // JSON本文をテーブルに追加または上書きします。
    private func upsert<T: Encodable>(table: String, id: UUID, parentColumn: String? = nil, parentID: UUID? = nil, nullableColumn: String? = nil, nullableID: UUID? = nil, value: T) throws {
        // Codableの値を保存用JSON文字列へ変換します。
        guard let json = String(data: try encoder.encode(value), encoding: .utf8) else { throw LibraryDatabaseError.encoding("JSONをUTF-8文字列に変換できません") }
        // 顔テーブルでは動画IDと任意の人物IDを追加で保存します。
        if table == "faces" {
            // 顔レコード用のupsert文を実行します。
            try execute("INSERT INTO faces(id, video_id, person_id, payload) VALUES ('\(id.uuidString)', '\(parentID!.uuidString)', \(nullableID.map { "'\($0.uuidString)'" } ?? "NULL"), '\(escape(json))') ON CONFLICT(id) DO UPDATE SET video_id=excluded.video_id, person_id=excluded.person_id, payload=excluded.payload")
        // 動画テーブルには所属フォルダIDも保存します。
        } else if table == "videos" {
            // 動画レコード用のupsert文を実行します。
            try execute("INSERT INTO videos(id, source_id, payload) VALUES ('\(id.uuidString)', '\(parentID!.uuidString)', '\(escape(json))') ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id, payload=excluded.payload")
        // フォルダや人物には親ID列がありません。
        } else {
            // 単純なIDとJSON本文のupsert文を実行します。
            try execute("INSERT INTO \(table)(id, payload) VALUES ('\(id.uuidString)', '\(escape(json))') ON CONFLICT(id) DO UPDATE SET payload=excluded.payload")
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }
    // テーブルのJSON本文をすべてデコードして返します。
    private func readAll<T: Decodable>(_ table: String, as type: T.Type) throws -> [T] {
        // JSON列を取得するSQLを準備します。
        var statement: OpaquePointer?
        // SQLをSQLite文へ変換します。
        guard sqlite3_prepare_v2(handle, "SELECT payload FROM \(table) ORDER BY rowid", -1, &statement, nil) == SQLITE_OK else { throw makeError() }
        // 文を最後に解放します。
        defer { sqlite3_finalize(statement) }
        // 復元したレコードを入れる配列です。
        var values: [T] = []
        // 取得行がなくなるまで1行ずつ読みます。
        while sqlite3_step(statement) == SQLITE_ROW {
            // SQLiteのUTF-8文字列をSwift文字列に変換します。
            guard let text = sqlite3_column_text(statement, 0) else { throw makeError() }
            // JSON文字列を指定型へ復元して配列に追加します。
            values.append(try decoder.decode(type, from: Data(bytes: text, count: Int(sqlite3_column_bytes(statement, 0)))))
        // ここで直前の処理の範囲を閉じます。
        }
        // 全件を返します。
        return values
    // ここで直前の処理の範囲を閉じます。
    }
    // 指定動画に保存済みの顔が参照する人物 ID を読み取ります。
    private func personIDsForVideo(_ videoID: UUID) throws -> Set<UUID> {
        // 人物 ID の重複を除いた結果を格納します。
        var identifiers = Set<UUID>()
        // 検索する SQL 文を保持します。
        var statement: OpaquePointer?
        // UUID はアプリ内部で生成した値だけを SQL に渡します。
        let sql = "SELECT DISTINCT person_id FROM faces WHERE video_id = '\(videoID.uuidString)' AND person_id IS NOT NULL"
        // SQL を実行可能な文に変換します。
        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else { throw makeError() }
        // 読み取り後には文を解放します。
        defer { sqlite3_finalize(statement) }
        // 結果が残っている間は人物 ID を読み取ります。
        while sqlite3_step(statement) == SQLITE_ROW {
            // SQLite が保持する文字列を取り出します。
            guard let value = sqlite3_column_text(statement, 0) else { throw makeError() }
            // UUID として復元できた人物だけを候補に加えます。
            if let identifier = UUID(uuidString: String(cString: value)) { identifiers.insert(identifier) }
        // ここで直前の処理の範囲を閉じます。
        }
        // 候補となる人物 ID を返します。
        return identifiers
    // ここで直前の処理の範囲を閉じます。
    }
    // テーブルからIDで指定した行を削除します。
    private func delete(table: String, id: UUID) throws {
        // 外部キー規則に従って削除します。
        try execute("DELETE FROM \(table) WHERE id = '\(id.uuidString)'")
    // ここで直前の処理の範囲を閉じます。
    }
    // SQL文を実行して失敗した場合はエラーを返します。
    private func execute(_ sql: String) throws {
        // SQLiteが必要とするC文字列としてSQLを渡します。
        let result = sqlite3_exec(handle, sql, nil, nil, nil)
        // 成功以外の結果を説明付きエラーにします。
        guard result == SQLITE_OK else { throw makeError() }
    // ここで直前の処理の範囲を閉じます。
    }
    // JSON内のシングルクォートをSQL文字列用に二重化します。
    private func escape(_ value: String) -> String {
        // SQLの文字列リテラルで安全な表記に置き換えます。
        return value.replacingOccurrences(of: "'", with: "''")
    // ここで直前の処理の範囲を閉じます。
    }
    // SQLiteの現在のエラー内容をSwiftのエラーへ変換します。
    private func makeError() -> LibraryDatabaseError {
        // 接続がない場合にも説明を返します。
        return .sqlite(handle.map { String(cString: sqlite3_errmsg($0)) } ?? "SQLite接続がありません")
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
