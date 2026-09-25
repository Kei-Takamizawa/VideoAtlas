// アプリの状態を画面へ通知するために Observation を読み込みます。
import Observation
// ファイル、日付、URL を扱うために Foundation を読み込みます。
import Foundation

// 解析の一時停止指示を別の実行文脈から安全に読むための箱です。
private final class ScanStopFlag: @unchecked Sendable {
    // 複数の実行文脈から値を守るロックです。
    private let lock = NSLock()
    // 停止指示の保存値です。
    private var stored = false
    // 現在の停止指示を返します。
    var value: Bool {
        // 読み取り中は値を固定します。
        lock.lock()
        // 関数を抜ける際にロックを解放します。
        defer { lock.unlock() }
        // 保存された値を返します。
        return stored
    // ここで直前の処理の範囲を閉じます。
    }
    // 停止指示を書き込みます。
    func set(_ newValue: Bool) {
        // 書き込み中は値を固定します。
        lock.lock()
        // 新しい指示を保存します。
        stored = newValue
        // ロックを解放します。
        lock.unlock()
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// 高頻度の解析通知から画面描画に必要な分だけを通す箱です。
private final class ScanProgressGate: @unchecked Sendable {
    // 最後の通知時刻を守るロックです。
    private let lock = NSLock()
    // 最後に画面へ通知した時刻です。
    private var lastPublished = 0.0
    // 約四分の一秒に一度だけ画面へ通知します。
    func shouldPublish() -> Bool {
        // 一度に一つの通知だけが時刻を変更します。
        lock.lock()
        // 関数を抜けるときロックを解除します。
        defer { lock.unlock() }
        // 現在の単調増加時計を読みます。
        let now = ProcessInfo.processInfo.systemUptime
        // 前回から十分に時間が経っていなければ通知を省きます。
        guard now - lastPublished >= 0.25 else { return false }
        // 今回の時刻を次回の判定に残します。
        lastPublished = now
        // 画面に進捗を反映してよいことを伝えます。
        return true
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// 画面表示、動画解析、SQLite 保存をまとめる主実行文脈のモデルです。
@MainActor @Observable final class AppModel {
    // 登録したフォルダの一覧です。
    private(set) var sources: [SourceFolder] = []
    // 発見した動画の一覧です。
    private(set) var videos: [VideoRecord] = []
    // 検出した顔の一覧です。
    private(set) var faces: [FaceRecord] = []
    // 人物の一覧です。
    private(set) var people: [PersonRecord] = []
    // 解析が実行中なら true です。
    private(set) var isScanning = false
    // 画面に表示する現在の処理内容です。
    private(set) var statusText = ""
    // 画面の進捗率を 0 から 1 で表します。
    private(set) var progress = 0.0
    // SQLite 接続を保持します。
    @ObservationIgnored private var database: LibraryDatabase?
    // 背景の保存処理が別接続を開くための保存先です。
    @ObservationIgnored private var databaseURL: URL?
    // サムネイルの保存先を保持します。
    @ObservationIgnored private var thumbnailDirectory: URL?
    // 現在の走査処理を保持します。
    @ObservationIgnored private var scanTask: Task<Void, Never>?
    // 解析中だけ使う動画ごとの顔索引です。
    @ObservationIgnored private var scanFacesByVideo: [UUID: [FaceRecord]] = [:]
    // 解析中だけ使う人物ごとの少数の比較用顔です。
    @ObservationIgnored private var scanRepresentatives: [UUID: [FaceRecord]] = [:]
    // 一時停止の状態を解析器へ知らせます。
    @ObservationIgnored private let stopFlag = ScanStopFlag()
    // 明示的な再解析で使うサンプル間隔です。
    private let standardInterval = 2.0
    // 詳細な再解析で使うサンプル間隔です。
    private let detailedInterval = 0.5

    // アプリ専用の保存先を作り、以前のライブラリを読み込みます。
    init(storageDirectory: URL? = nil) {
        // Application Support の標準ディレクトリを取得します。
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        // このアプリ専用のサブディレクトリを指定します。
        let directory = storageDirectory ?? support.appendingPathComponent("VideoAtlas", isDirectory: true)
        // サムネイル保存先を指定します。
        let thumbnails = directory.appendingPathComponent("Thumbnails", isDirectory: true)
        // ディレクトリとデータベースの準備を試します。
        do {
            // アプリの保存先を作ります。
            try FileManager.default.createDirectory(at: thumbnails, withIntermediateDirectories: true)
            // 生成した画像の保存先を保持します。
            thumbnailDirectory = thumbnails
            // SQLite データベースを開きます。
            databaseURL = directory.appendingPathComponent("library.sqlite")
            // 画面側の読み書き用に SQLite 接続を開きます。
            database = try LibraryDatabase(databaseURL: databaseURL!)
            // 保存済みレコードを読み込みます。
            try reload()
            // 前回終了時に解析中だった動画を再開待ちに戻します。
            for index in videos.indices where videos[index].state == .analyzing {
                // 解析が中断されたことを状態に反映します。
                videos[index].state = .pending
                // 再開可能な時刻を残したまま保存します。
                try database?.saveVideo(videos[index])
            // ここで直前の処理の範囲を閉じます。
            }
            // 現行版の自動統合を既に実行したか調べます。
            let alreadyReconciled = try database?.hasCurrentGroupReconciliation() ?? true
            // 既存の解析済みライブラリへ新しい統合規則を一度だけ適用します。
            if !videos.contains(where: { $0.state == .pending }) && videos.filter({ $0.state == .ready }).count >= 2 && !alreadyReconciled {
                // 背景処理中は手動修正との競合を防ぎます。
                isScanning = true
                // 起動直後の処理内容を画面へ伝えます。
                statusText = "既存の人物グループを照合中…"
                // 起動を妨げずに既存ライブラリの一度限りの照合を始めます。
                scanTask = Task { await reconcileExistingLibrary() }
            // ここで直前の処理の範囲を閉じます。
            }
        // 保存先やデータベースを使えない場合は画面へ理由を表示します。
        } catch {
            // 利用者が対処できるよう原因を残します。
            statusText = "ライブラリを開けません：\(error.localizedDescription)"
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // フォルダの権限を保存し、すぐに一覧を更新します。
    func addFolder(url: URL) {
        // ブックマーク作成と保存を試します。
        do {
            // 後の起動でもフォルダを開ける許可情報を作ります。
            let bookmark = try FolderAccess.createBookmark(url: url)
            // 同じフォルダが既にあれば権限情報だけ更新します。
            if let index = sources.firstIndex(where: { $0.path == url.standardizedFileURL.path }) {
                // 新しく選んだフォルダのブックマークへ置き換えます。
                sources[index].bookmark = bookmark
                // 再起動後にも新しい読み取り権限を使えるよう保存します。
                try database?.saveSource(sources[index])
                // 権限不足で失敗した動画を再解析待ちへ戻します。
                for videoIndex in videos.indices where videos[videoIndex].sourceID == sources[index].id && videos[videoIndex].state == .failed {
                    // 再度読めるか確認するため未処理にします。
                    videos[videoIndex].state = .pending
                    // 古い権限エラーを消します。
                    videos[videoIndex].errorMessage = nil
                    // 再試行の状態を保存します。
                    try database?.saveVideo(videos[videoIndex])
                // ここで直前の処理の範囲を閉じます。
                }
                // 更新した権限で動画を探し直します。
                refreshWhenAvailable()
                // 同じフォルダの二重登録は行いません。
                return
            // ここで直前の処理の範囲を閉じます。
            }
            // 一意の ID を持つ登録情報を作ります。
            let source = SourceFolder(id: UUID(), path: url.standardizedFileURL.path, bookmark: bookmark)
            // SQLite に保存します。
            try database?.saveSource(source)
            // 画面の一覧へ追加します。
            sources.append(source)
            // 新しいフォルダ内の動画を調べます。
            refreshWhenAvailable()
        // 権限取得や保存に失敗した場合は理由を表示します。
        } catch {
            // 発生した原因を画面に残します。
            statusText = "フォルダを登録できません：\(error.localizedDescription)"
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // 登録フォルダを列挙し、変更された動画だけを解析待ちにします。
    func refreshLibrary() {
        // 解析中は一覧更新を重ねません。
        guard !isScanning else { return }
        // フォルダごとに権限を取得します。
        for source in sources {
            // ブックマークの解決とファイル列挙を試します。
            do {
                // 権限付きフォルダ URL を復元します。
                let folder = try FolderAccess.resolve(source)
                // 利用後にセキュリティスコープを必ず閉じます。
                defer { folder.stopAccessingSecurityScopedResource() }
                // フォルダが存在することを確認します。
                guard FileManager.default.fileExists(atPath: folder.path) else { throw CocoaError(.fileNoSuchFile) }
                // 対象の動画ファイルを再帰的に探します。
                let urls = try FolderAccess.enumerateVideos(in: folder)
                // 見つかったパスを記録します。
                let found = Set(urls.map { $0.standardizedFileURL.path })
                // 元フォルダに登録済みの動画の存在を更新します。
                for index in videos.indices where videos[index].sourceID == source.id {
                    // ファイルが見つからなかった動画を欠落扱いにします。
                    if !found.contains(videos[index].path) {
                        // 欠落状態を記録します。
                        videos[index].state = .missing
                        // 欠落の原因を説明します。
                        videos[index].errorMessage = "登録フォルダ内で元動画が見つかりません。"
                        // 状態を永続化します。
                        try database?.saveVideo(videos[index])
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
                // 見つかったファイルの更新時刻とサイズを調べます。
                for url in urls {
                    // ファイル属性を取得します。
                    let values = try url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
                    // 更新時刻が読めないファイルは明示的に失敗させます。
                    guard let modified = values.contentModificationDate, let size = values.fileSize else { throw CocoaError(.fileReadUnknown) }
                    // 同じ元フォルダとパスの既存動画を探します。
                    if let index = videos.firstIndex(where: { $0.sourceID == source.id && $0.path == url.standardizedFileURL.path }) {
                        // サイズまたは更新日時が違う場合だけ旧解析を無効にします。
                        let contentChanged = videos[index].fileSize != Int64(size) || videos[index].modificationTime != modified
                        // 内容変更か前回欠落だった場合は再解析を予約します。
                        if contentChanged || videos[index].state == .missing {
                            // 保存完了までは画面上の旧状態を変更しないため複製します。
                            var updated = videos[index]
                            // 最新サイズを記録します。
                            updated.fileSize = Int64(size)
                            // 最新時刻を記録します。
                            updated.modificationTime = modified
                            // 未処理にします。
                            updated.state = .pending
                            // 以前のエラーを消します。
                            updated.errorMessage = nil
                            // ファイルの中身が変わった場合は旧顔と旧ポスターを使いません。
                            if contentChanged {
                                // 新しい映像は先頭から解析します。
                                updated.lastAnalyzedSecond = 0
                                // 古い動画の長さを表示しません。
                                updated.duration = 0
                                // 古い動画の代表画像を参照しません。
                                updated.posterPath = nil
                                // 動画情報と旧顔削除を一つのトランザクションで保存します。
                                let removedPeople = try database?.invalidateVideoRevision(updated) ?? []
                                // 保存できた新しい動画情報を画面へ反映します。
                                videos[index] = updated
                                // 旧内容で検出した顔を画面から取り除きます。
                                faces.removeAll { $0.videoID == updated.id }
                                // 他の動画から参照されない人物だけを画面から取り除きます。
                                people.removeAll { removedPeople.contains($0.id) }
                            // ファイルが同じまま戻った場合は手動修正を残します。
                            } else {
                                // 欠落状態だけを解析待ちへ変更します。
                                try database?.saveVideo(updated)
                                // 保存できた状態を画面へ反映します。
                                videos[index] = updated
                            // ここで直前の処理の範囲を閉じます。
                            }
                        // ここで直前の処理の範囲を閉じます。
                        }
                    // 新しい動画には新しい ID を付けます。
                    } else {
                        // 初期状態の動画レコードを作ります。
                        let video = VideoRecord(id: UUID(), sourceID: source.id, path: url.standardizedFileURL.path, name: url.lastPathComponent, duration: 0, fileSize: Int64(size), modificationTime: modified, state: .pending, lastAnalyzedSecond: 0, sampleInterval: standardInterval, errorMessage: nil, posterPath: nil)
                        // 動画レコードを永続化します。
                        try database?.saveVideo(video)
                        // 一覧へ追加します。
                        videos.append(video)
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
            // フォルダごとの失敗を表示し、他のフォルダは続けます。
            } catch {
                // アクセス失敗時には欠落と断定しません。
                statusText = "\(source.path) を確認できません：\(error.localizedDescription)"
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 未処理動画があれば解析を開始します。
        resumeScanning()
    // ここで直前の処理の範囲を閉じます。
    }

    // 解析中に選ばれたフォルダも処理終了後に必ず列挙します。
    private func refreshWhenAvailable() {
        // 別の解析処理が動いていなければ直ちに動画を探します。
        guard isScanning else { refreshLibrary(); return }
        // 現在動いている処理の参照を固定します。
        let runningTask = scanTask
        // 処理が終わった時点で新しいフォルダを探します。
        Task {
            // 動画解析や人物照合の終了を待ちます。
            await runningTask?.value
            // 後から登録されたフォルダの動画を列挙します。
            refreshLibrary()
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // 動画解析に停止を依頼し、途中結果を保存させます。
    func pauseScanning() {
        // 現在の動画処理へ停止を通知します。
        stopFlag.set(true)
        // 画面には保存完了まで処理中と伝えます。
        statusText = "一時停止して途中結果を保存しています…"
    // ここで直前の処理の範囲を閉じます。
    }

    // 解析待ち動画の処理を開始します。
    func resumeScanning() {
        // 二重起動や保存先なしの実行を防ぎます。
        guard !isScanning, database != nil, thumbnailDirectory != nil else { return }
        // 解析対象の有無を確認します。
        guard videos.contains(where: { $0.state == .pending }) else { return }
        // 前回の停止指示を解除します。
        stopFlag.set(false)
        // 全顔を毎区間調べずに済むよう動画 ID ごとに分類します。
        scanFacesByVideo = Dictionary(grouping: faces, by: \.videoID)
        // 欠落や失敗の動画から人物を推測しないよう有効な動画を選びます。
        let eligibleVideoIDs = Set(videos.filter { $0.state != .missing && $0.state != .failed }.map(\.id))
        // 前回の解析で使った比較用の顔を作り直します。
        scanRepresentatives = [:]
        // 有効な顔から人物ごとの多様な代表だけを選びます。
        for face in faces where !face.excluded && eligibleVideoIDs.contains(face.videoID) {
            // 似た顔を重複させず多様な姿勢の顔を代表にします。
            FaceBatchStore.addRepresentative(face, to: &scanRepresentatives)
        // ここで直前の処理の範囲を閉じます。
        }
        // 実行中の表示へ切り替えます。
        isScanning = true
        // 解析用の非同期処理を開始します。
        scanTask = Task { await scanPendingVideos() }
    // ここで直前の処理の範囲を閉じます。
    }

    // 指定した動画を再解析待ちにします。
    func rescan(videoID: UUID, detailed: Bool) {
        // 指定 ID の動画を探します。
        guard let index = videos.firstIndex(where: { $0.id == videoID }) else { return }
        // 現在解析中の動画なら、途中結果の保存完了後に再解析を予約します。
        if videos[index].state == .analyzing {
            // 現在の解析に中断を知らせます。
            stopFlag.set(true)
            // 保存が終わってから先頭からの再解析を設定します。
            Task {
                // 今の解析処理の終了を待ちます。
                await scanTask?.value
                // 同じ ID がまだ残っていれば再解析へ進みます。
                if videos.contains(where: { $0.id == videoID }) { rescan(videoID: videoID, detailed: detailed) }
            // ここで直前の処理の範囲を閉じます。
            }
            // 終了待ちの間は既存の状態を維持します。
            return
        // ここで直前の処理の範囲を閉じます。
        }
        // 解析済み位置を最初へ戻します。
        videos[index].lastAnalyzedSecond = 0
        // 詳細指定に応じて間隔を設定します。
        videos[index].sampleInterval = detailed ? detailedInterval : standardInterval
        // 解析待ちにします。
        videos[index].state = .pending
        // 前回のエラーを消します。
        videos[index].errorMessage = nil
        // 状態を保存し、可能なら解析を始めます。
        persistVideo(at: index)
        // 現在の走査後にも新しい予約を処理します。
        resumeScanning()
    // ここで直前の処理の範囲を閉じます。
    }

    // 人物の表示名を変更します。
    func renamePerson(id: UUID, name: String) {
        // 背景保存中の手動変更と自動割当が衝突しないようにします。
        guard !isScanning else { statusText = "解析を一時停止してから人物を編集してください。"; return }
        // 空白だけの名前は受け付けません。
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        // 指定人物を探します。
        guard let index = people.firstIndex(where: { $0.id == id }) else { return }
        // 表示名を更新します。
        people[index].name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        // 新しい名前を保存します。
        persist { try database?.savePerson(people[index]) }
    // ここで直前の処理の範囲を閉じます。
    }

    // 元人物のすべての顔を統合先へ手動割当します。
    func mergePeople(sourceID: UUID, targetID: UUID) {
        // 背景保存中の人物統合は一時停止後に行います。
        guard !isScanning else { statusText = "解析を一時停止してから人物を編集してください。"; return }
        // 同じ人物同士や存在しない人物の統合を防ぎます。
        guard sourceID != targetID, people.contains(where: { $0.id == sourceID }), people.contains(where: { $0.id == targetID }) else { return }
        // 割り当てられていた各顔を移動します。
        for index in faces.indices where faces[index].personID == sourceID {
            // 統合先へ付け替えます。
            faces[index].personID = targetID
            // 人手で決めた分類として固定します。
            faces[index].manualAssignment = true
            // 顔の変更を保存します。
            persist { try database?.saveFace(faces[index]) }
        // ここで直前の処理の範囲を閉じます。
        }
        // 統合元の人物を SQLite から削除します。
        persist { try database?.removePerson(id: sourceID) }
        // 画面の人物一覧からも取り除きます。
        people.removeAll { $0.id == sourceID }
    // ここで直前の処理の範囲を閉じます。
    }

    // 一つの顔を新しい人物へ手動で分離します。
    func splitFace(faceID: UUID) {
        // 背景保存中の人物分割は一時停止後に行います。
        guard !isScanning else { statusText = "解析を一時停止してから人物を編集してください。"; return }
        // 分割対象を探します。
        guard let index = faces.firstIndex(where: { $0.id == faceID }) else { return }
        // 表示名を付けずに新しい人物を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: faces[index].thumbnailPath)
        // 先に人物を保存します。
        persist { try database?.savePerson(person) }
        // 人物一覧へ追加します。
        people.append(person)
        // 対象顔を新しい人物へ割り当てます。
        faces[index].personID = person.id
        // 手動操作として記録します。
        faces[index].manualAssignment = true
        // 顔の変更を保存します。
        persist { try database?.saveFace(faces[index]) }
    // ここで直前の処理の範囲を閉じます。
    }

    // 顔を既存の人物または未分類へ手動で移します。
    func assignFace(faceID: UUID, to personID: UUID?) {
        // 背景保存中の人物割当は一時停止後に行います。
        guard !isScanning else { statusText = "解析を一時停止してから人物を編集してください。"; return }
        // 指定された顔の位置を探します。
        guard let index = faces.firstIndex(where: { $0.id == faceID }) else { return }
        // 統合先がある場合は人物一覧に存在することを確認します。
        guard personID == nil || people.contains(where: { $0.id == personID }) else { return }
        // 顔の人物割当を変更します。
        faces[index].personID = personID
        // この割当は人が決めたものとして固定します。
        faces[index].manualAssignment = true
        // 再分類された顔は再び一覧で見られるようにします。
        faces[index].excluded = false
        // 手動変更をデータベースに保存します。
        persist { try database?.saveFace(faces[index]) }
    // ここで直前の処理の範囲を閉じます。
    }

    // 顔を人物一覧から除外します。
    func excludeFace(faceID: UUID) {
        // 背景保存中の除外操作は一時停止後に行います。
        guard !isScanning else { statusText = "解析を一時停止してから人物を編集してください。"; return }
        // 対象顔を探します。
        guard let index = faces.firstIndex(where: { $0.id == faceID }) else { return }
        // 除外フラグを設定します。
        faces[index].excluded = true
        // 自動分類が戻さないよう手動操作として固定します。
        faces[index].manualAssignment = true
        // 顔を保存します。
        persist { try database?.saveFace(faces[index]) }
    // ここで直前の処理の範囲を閉じます。
    }

    // 登録情報と生成した画像を消去します。
    func clearIndex() {
        // 実行中の解析を止めるよう指示します。
        stopFlag.set(true)
        // 現在の処理が終了してからデータを消します。
        Task {
            // 実行中の走査タスクを待ちます。
            await scanTask?.value
            // データベースの削除が成功したときだけ画像と画面を消します。
            do {
                // 保存先が使えることを確認します。
                guard let database else { throw LibraryDatabaseError.encoding("データベースを開けません") }
                // 顔・動画・フォルダの索引を一括で消します。
                try database.clearAll()
            // データベースを消せない場合は生成画像を残します。
            } catch {
                // 利用者へ削除できなかった理由を表示します。
                statusText = "インデックスを消去できません：\(error.localizedDescription)"
                // 画面のデータはそのまま保持します。
                return
            // ここで直前の処理の範囲を閉じます。
            }
            // 生成画像だけを削除します。
            if let directory = thumbnailDirectory {
                // 保存先配下の項目を列挙します。
                for url in (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [] {
                    // 個別の生成画像を削除します。
                    try? FileManager.default.removeItem(at: url)
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
            // 画面の登録フォルダを空にします。
            sources = []
            // 画面の動画を空にします。
            videos = []
            // 画面の顔を空にします。
            faces = []
            // 画面の人物を空にします。
            people = []
            // 進捗を初期化します。
            progress = 0
            // 完了を表示します。
            statusText = "インデックスを消去しました。"
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // 登録フォルダとその動画・顔を削除します。
    func removeSource(id: UUID) {
        // 背景処理が参照する登録元は解析終了後に削除します。
        guard !isScanning else { statusText = "解析を一時停止してからフォルダを削除してください。"; return }
        // 対象フォルダに属する動画 ID を集めます。
        let videoIDs = Set(videos.filter { $0.sourceID == id }.map(\.id))
        // データベースの連鎖削除を実行します。
        persist { try database?.removeSource(id: id) }
        // 登録フォルダ一覧を更新します。
        sources.removeAll { $0.id == id }
        // 顔一覧を更新します。
        faces.removeAll { videoIDs.contains($0.videoID) }
        // 動画一覧を更新します。
        videos.removeAll { $0.sourceID == id }
    // ここで直前の処理の範囲を閉じます。
    }

    // SQLite から最新の登録情報を読み込みます。
    private func reload() throws {
        // 開いた接続から全テーブルを復元します。
        guard let snapshot = try database?.load() else { return }
        // フォルダ一覧を反映します。
        sources = snapshot.sources
        // 動画一覧を反映します。
        videos = snapshot.videos
        // 顔一覧を反映します。
        faces = snapshot.faces
        // 人物一覧を反映します。
        people = snapshot.people
    // ここで直前の処理の範囲を閉じます。
    }

    // 保存エラーを画面へ通知します。
    private func persist(_ operation: () throws -> Void) {
        // SQLite の操作を試します。
        do {
            // 渡された保存処理を実行します。
            try operation()
        // 失敗した場合は原因を表示します。
        } catch {
            // エラー内容を画面へ表示します。
            statusText = "保存できません：\(error.localizedDescription)"
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // 指定位置の動画を永続化します。
    private func persistVideo(at index: Int) {
        // 現在のレコードを SQLite へ書き込みます。
        persist { try database?.saveVideo(videos[index]) }
    // ここで直前の処理の範囲を閉じます。
    }

    // 以前に解析済みのライブラリへ現行版の統合を一度だけ適用します。
    private func reconcileExistingLibrary() async {
        // 背景の人物比較と SQLite 保存を試します。
        do {
            // 現在の解析結果に対して慎重な自動統合を実行します。
            let output = try await performGroupReconciliation()
            // 実際に統合した件数を表示します。
            statusText = "既存ライブラリの照合完了：人物グループを \(output.personMapping.count) 件統合しました。"
        // 記録や比較に失敗しても保存済みの顔を消しません。
        } catch {
            // 自動統合だけが失敗した理由を画面へ表示します。
            statusText = "人物グループを照合できません：\(error.localizedDescription)"
        // ここで直前の処理の範囲を閉じます。
        }
        // 解析中の表示を解除します。
        isScanning = false
        // 既存の動画はすべて解析済みなので進捗を完了へ揃えます。
        progress = 1
        // 完了した作業の参照を外します。
        scanTask = nil
    // ここで直前の処理の範囲を閉じます。
    }

    // 顔の写しを背景で比較し、確定した人物 ID だけ画面へ反映します。
    private func performGroupReconciliation() async throws -> GroupReconciliationOutput {
        // 背景用接続を開けることを確認します。
        guard let databaseURL, let database else { throw LibraryDatabaseError.encoding("データベースを開けません") }
        // 手動修正を含む現在のライブラリを値の写しにします。
        let input = GroupReconciliationInput(faces: faces, videos: videos, people: people, databaseURL: databaseURL)
        // 多数の特徴量比較と SQLite 更新を主実行文脈から切り離します。
        let output = try await Task.detached(priority: .utility) { try GroupReconciler.reconcile(input) }.value
        // 統合があれば画面の顔を一度の配列更新で反映します。
        if !output.personMapping.isEmpty {
            // 画面側の顔を値として複製します。
            var updatedFaces = faces
            // 統合元の人物 ID だけを最終人物へ変更します。
            for index in updatedFaces.indices {
                // この顔が統合元なら新しい人物 ID を設定します。
                if let personID = updatedFaces[index].personID, let destination = output.personMapping[personID] { updatedFaces[index].personID = destination }
            // ここで直前の処理の範囲を閉じます。
            }
            // 人物による絞り込みへ確定済みの顔を反映します。
            faces = updatedFaces
            // 統合元の顔写真タイルを除きます。
            people.removeAll { output.personMapping[$0.id] != nil }
        // ここで直前の処理の範囲を閉じます。
        }
        // 既存ライブラリで同じ重い照合を繰り返さないよう版を保存します。
        try database.markCurrentGroupReconciliation()
        // 変更件数を呼び出し元へ返します。
        return output
    // ここで直前の処理の範囲を閉じます。
    }

    // 解析待ち動画を順番に処理します。
    private func scanPendingVideos() async {
        // 全動画終了後の人物統合結果を表示するため保持します。
        var reconciliationSummary: String?
        // 解析器の準備を試します。
        do {
            // OpenVINO を含む解析器を用意します。
            let analyzer = try await Task.detached(priority: .utility) { try VideoAnalyzer() }.value
            // 処理済み件数を数えます。
            var processed = 0
            // 頻繁すぎる進捗通知を抑える状態です。
            let progressGate = ScanProgressGate()
            // 開始時点の対象件数を記録します。
            let total = max(1, videos.filter { $0.state == .pending }.count)
            // 未処理動画が残る間に処理します。
            while !stopFlag.value, let index = videos.firstIndex(where: { $0.state == .pending }) {
                // 現在の動画の ID と登録元を保持します。
                let videoID = videos[index].id
                // 削除や配列更新の前にフォルダを探します。
                guard let source = sources.first(where: { $0.id == videos[index].sourceID }) else {
                    // 所属フォルダがない動画は失敗として残します。
                    videos[index].state = .failed
                    // 必要なフォルダがない理由を表示します。
                    videos[index].errorMessage = "登録元のフォルダがありません。"
                    // 次回起動後も原因が分かるよう保存します。
                    persistVideo(at: index)
                    // 他の動画の解析は続けます。
                    continue
                // ここで直前の処理の範囲を閉じます。
                }
                // 一つのフォルダだけに問題があっても他の動画を続けます。
                let folder: URL
                // ブックマークの解決を動画ごとに試します。
                do {
                    // この動画の登録元へ読み取り権限を取得します。
                    folder = try FolderAccess.resolve(source)
                // 権限が切れた場合はこの動画だけ失敗にします。
                } catch {
                    // 権限を使えない動画として記録します。
                    videos[index].state = .failed
                    // フォルダ再選択を案内する説明を残します。
                    videos[index].errorMessage = "フォルダの読み取り権限を確認できません。再選択してください：\(error.localizedDescription)"
                    // エラーをデータベースへ保存します。
                    persistVideo(at: index)
                    // 次の動画へ進みます。
                    processed += 1
                    // 他のフォルダの動画は解析を続けます。
                    continue
                // ここで直前の処理の範囲を閉じます。
                }
                // この一件の処理範囲を明示します。
                do {
                // 一件を処理し終えたらアクセスを必ず解放します。
                defer { folder.stopAccessingSecurityScopedResource() }
                // 動画のファイル URL を構築します。
                let url = URL(fileURLWithPath: videos[index].path)
                // 元ファイルがなければ欠落状態へ移します。
                guard FileManager.default.fileExists(atPath: url.path) else {
                    // 欠落状態へ更新します。
                    videos[index].state = .missing
                    // エラー内容を保存します。
                    videos[index].errorMessage = "元動画が見つかりません。"
                    // 状態を保存します。
                    persistVideo(at: index)
                    // 次の動画へ進みます。
                    continue
                // ここで直前の処理の範囲を閉じます。
                }
                // 解析開始を記録します。
                videos[index].state = .analyzing
                // 状態を保存します。
                persistVideo(at: index)
                // 現在の動画名を表示します。
                statusText = "解析中：\(videos[index].name)"
                // 解析中に現在位置を画面へ伝えます。
                let result: VideoAnalysisResult
                // 並行実行の通知で変更されないよう処理済み件数を固定します。
                let completedBefore = processed
                // 個別動画の失敗は次の動画に進めるよう捕捉します。
                do {
                    // 保存済みの位置から顔解析を進めます。
                    result = try await analyzer.analyze(url: url, interval: videos[index].sampleInterval, startingAt: videos[index].lastAnalyzedSecond, endingAt: videos[index].lastAnalyzedSecond + videos[index].sampleInterval * 60, progress: { [weak self] second, duration in
                        // 進捗表示だけを主実行文脈へ渡します。
                        if progressGate.shouldPublish() { Task { @MainActor [weak self] in self?.progress = min(1, (Double(completedBefore) + second / max(duration, 1)) / Double(total)) } }
                    // ここで直前の処理の範囲を閉じます。
                    }, isCancelled: { [stopFlag] in stopFlag.value })
                // この動画で解析エラーが起きた場合です。
                } catch {
                    // 動画が残っていれば失敗情報を書きます。
                    if let current = videos.firstIndex(where: { $0.id == videoID }) {
                        // 失敗状態にします。
                        videos[current].state = .failed
                        // 原因を保存します。
                        videos[current].errorMessage = error.localizedDescription
                        // 状態を永続化します。
                        persistVideo(at: current)
                    // ここで直前の処理の範囲を閉じます。
                    }
                    // 次の動画へ進みます。
                    processed += 1
                    // 失敗の要約を画面に表示します。
                    statusText = "解析失敗：\(url.lastPathComponent)：\(error.localizedDescription)"
                    // 次の対象へ進みます。
                    continue
                // ここで直前の処理の範囲を閉じます。
                }
                // 処理中にフォルダが削除された場合は結果を使いません。
                guard let current = videos.firstIndex(where: { $0.id == videoID }) else { continue }
                // サムネイル保存の失敗を動画単位で処理します。
                do {
                    // サムネイルを保存しながら顔結果を統合します。
                    try await accept(result: result, videoIndex: current)
                // 保存の失敗時も次の動画を処理できるようにします。
                } catch {
                    // 失敗状態へ移します。
                    videos[current].state = .failed
                    // 失敗した理由を表示用に保存します。
                    videos[current].errorMessage = "解析結果を保存できません：\(error.localizedDescription)"
                    // 動画状態を永続化します。
                    persistVideo(at: current)
                    // 画面にも失敗を表示します。
                    statusText = videos[current].errorMessage ?? "解析結果を保存できません。"
                    // 次の動画へ進みます。
                    processed += 1
                    // 同じ動画を繰り返し処理しません。
                    continue
                // ここで直前の処理の範囲を閉じます。
                }
                // 動画全体が終わった場合だけ完了した件数を更新します。
                if result.completedSecond >= result.duration { processed += 1 }
                // 中断されたら次の動画へ進まずループを抜けます。
                if result.wasCancelled { break }
                // 一件のアクセス範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
        // 解析器やブックマークの失敗を表示します。
        } catch {
            // 未処理の動画は残したまま原因を知らせます。
            statusText = "解析を続けられません：\(error.localizedDescription)"
        // ここで直前の処理の範囲を閉じます。
        }
        // 一時停止しておらず全動画を処理し終えたときだけ人物を照合します。
        if !stopFlag.value && !videos.contains(where: { $0.state == .pending }) && videos.filter({ $0.state == .ready }).count >= 2 {
            // 顔写真一覧の統合中であることを画面に伝えます。
            statusText = "人物グループを照合中…"
            // 重い比較と保存を背景で一度だけ実行します。
            do {
                // 動画をまたぐ顔の十分な一致を探します。
                let output = try await performGroupReconciliation()
                // 実際にまとめられた人物の数を記録します。
                reconciliationSummary = "解析完了：人物グループを \(output.personMapping.count) 件統合しました。"
            // 統合に失敗しても個々の動画の解析結果は残します。
            } catch {
                // 利用者が後で原因を確認できるよう残します。
                reconciliationSummary = "動画解析は完了しましたが、人物グループの照合に失敗しました：\(error.localizedDescription)"
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 実行フラグを解除します。
        isScanning = false
        // タスク参照を外します。
        scanTask = nil
        // 画面にある顔とは別の解析専用索引を解放します。
        scanFacesByVideo = [:]
        // 比較用の顔の写しも解放します。
        scanRepresentatives = [:]
        // 中断状態に応じて表示を更新します。
        if stopFlag.value { statusText = "一時停止しました。" }
        // 中断していなければ完了を表示します。
        else if !videos.contains(where: { $0.state == .pending }) { statusText = reconciliationSummary ?? (videos.contains(where: { $0.state == .failed }) ? "解析終了：失敗した動画の理由をカードに表示しています。" : "解析が完了しました。"); progress = 1 }
    // ここで直前の処理の範囲を閉じます。
    }

    // 一区間の顔を背景で保存し、画面には確定済みの少量だけを反映します。
    private func accept(result: VideoAnalysisResult, videoIndex: Int) async throws {
        // 保存先が使えない場合は動画に保存エラーを返します。
        guard let databaseURL, let thumbnailDirectory else { throw LibraryDatabaseError.encoding("保存先を開けません") }
        // 区間の開始時刻と既存の分類を値として固定します。
        let input = FaceBatchInput(result: result, video: videos[videoIndex], previousFaces: scanFacesByVideo[videos[videoIndex].id] ?? [], representativeFaces: scanRepresentatives.values.flatMap { $0 }, databaseURL: databaseURL, thumbnailDirectory: thumbnailDirectory)
        // 画像書き込み、顔比較、SQLite 更新を主実行文脈から切り離します。
        let output = try await Task.detached(priority: .utility) { try FaceBatchStore.accept(input) }.value
        // 保存中に動画が削除されていれば画面へ結果を追加しません。
        guard let current = videos.firstIndex(where: { $0.id == output.video.id }) else { return }
        // 再解析した区間の旧顔だけを画面から除きます。
        faces.removeAll { output.removedFaceIDs.contains($0.id) }
        // 確定済みの新しい顔を画面へ追加します。
        faces.append(contentsOf: output.faces)
        // 動画別の解析専用索引から古い区間を取り除きます。
        scanFacesByVideo[output.video.id, default: []].removeAll { output.removedFaceIDs.contains($0.id) }
        // 動画別の解析専用索引へ新しい顔を追加します。
        scanFacesByVideo[output.video.id, default: []].append(contentsOf: output.faces)
        // 古い比較用顔が再解析された区間にあれば除きます。
        for personID in Array(scanRepresentatives.keys) { scanRepresentatives[personID]?.removeAll { output.removedFaceIDs.contains($0.id) } }
        // 今回保存した顔から多様な比較用顔を補います。
        for face in output.faces where !face.excluded {
            // 同じ人物の異なる顔向きだけを代表に加えます。
            FaceBatchStore.addRepresentative(face, to: &scanRepresentatives)
        // ここで直前の処理の範囲を閉じます。
        }
        // 新しくできた人物を顔写真一覧へ追加します。
        people.append(contentsOf: output.people)
        // 動画の長さ、再開位置、状態、ポスターを画面へ反映します。
        videos[current] = output.video
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// 背景作業へ渡す値型の解析情報をまとめます。
struct FaceBatchInput: @unchecked Sendable {
    // 今回の顔検出結果です。
    let result: VideoAnalysisResult
    // 再開位置を含む動画の保存情報です。
    let video: VideoRecord
    // 手動編集と既存人物の比較に使う顔の読み取り専用写しです。
    let previousFaces: [FaceRecord]
    // 人物ごとに最大八枚だけの比較用顔です。
    let representativeFaces: [FaceRecord]
    // 背景専用の SQLite 接続を開く場所です。
    let databaseURL: URL
    // アプリ内で生成する顔画像の保存場所です。
    let thumbnailDirectory: URL
// ここで直前の処理の範囲を閉じます。
}

// 背景処理で保存が終わったレコードを画面へ返します。
struct FaceBatchOutput: @unchecked Sendable {
    // 今回の区間を反映した動画です。
    let video: VideoRecord
    // 今回の区間の新しい顔です。
    let faces: [FaceRecord]
    // 今回新しく作成した人物です。
    let people: [PersonRecord]
    // 再解析した区間から消す旧顔の ID です。
    let removedFaceIDs: Set<UUID>
// ここで直前の処理の範囲を閉じます。
}

// 顔比較と小さな区間の SQLite 保存を画面外で実行します。
enum FaceBatchStore {
    // 一区間の顔を前回の結果と照合して保存します。
    static func accept(_ input: FaceBatchInput) throws -> FaceBatchOutput {
        // 対象動画の一意な ID を短く保持します。
        let videoID = input.video.id
        // 同じ動画の以前の顔だけを取り出します。
        let previous = input.previousFaces
        // 今回の区間の終端を確認します。
        let end = input.result.completedSecond
        // 旧顔のうち今回再解析した時刻範囲だけを置き換えます。
        let oldRegion: [FaceRecord]
        // 映像を一枚も読まなかった区間では旧顔を変更しません。
        if end <= input.video.lastAnalyzedSecond { oldRegion = [] }
        // 画像生成器が要求時刻より最大 0.25 秒早い実フレームを返すことを考慮します。
        else {
            // 今回の最初の要求時刻より少し早い実フレームも対象にします。
            let lower = max(0, input.video.lastAnalyzedSecond - 0.25)
            // 通常区間では次区間の最初の実フレームを含まない位置で区切ります。
            let upper = end >= input.result.duration ? input.result.duration + 0.25 : end - 0.25
            // 再解析した実フレームの時刻帯だけ旧顔を置き換えます。
            oldRegion = previous.filter { $0.second >= lower && $0.second < upper }
        // ここで直前の処理の範囲を閉じます。
        }
        // 旧顔を実フレーム時刻ごとに探せる辞書へ変換します。
        var previousByFrame: [Int: [FaceRecord]] = [:]
        // 同じ時刻にある複数の顔も辞書に保持します。
        for face in oldRegion { previousByFrame[Int((face.second * 600).rounded()), default: []].append(face) }
        // 人物ごとに最大八件の多様な代表顔だけを保持します。
        var representatives: [UUID: [FaceRecord]] = [:]
        // 再解析する区間の旧顔が自身の分類を誘導しないようにします。
        let oldIDs = Set(oldRegion.map(\.id))
        // 他動画と処理済み区間の顔から比較用の代表を選びます。
        for face in input.representativeFaces where !face.excluded && !oldIDs.contains(face.id) && (face.videoID != videoID || face.second < input.video.lastAnalyzedSecond) {
            // 似た顔を重複させず代表集合へ加えます。
            addRepresentative(face, to: &representatives)
        // ここで直前の処理の範囲を閉じます。
        }
        // 同じフレームに一人の人物を二重登録しないための表です。
        var occupiedAtFrame: [Int: Set<UUID>] = [:]
        // 旧顔と同じ時刻の手動分類を保護します。
        for face in oldRegion where !face.excluded {
            // 人物が決まっている顔だけを占有情報へ追加します。
            if let personID = face.personID { occupiedAtFrame[Int((face.second * 600).rounded()), default: []].insert(personID) }
        // ここで直前の処理の範囲を閉じます。
        }
        // 同じ動画の直近フレームから時間的な追跡候補を作ります。
        var recentByPerson: [UUID: FaceRecord] = [:]
        // 通常解析と詳細解析のどちらでも少なくとも四秒半まで追跡します。
        let temporalGap = max(4.5, input.video.sampleInterval * 1.5)
        // 区間の開始前に保存済みの顔だけを追跡の起点にします。
        for face in previous where !face.excluded && !oldIDs.contains(face.id) && face.second < input.video.lastAnalyzedSecond {
            // 人物、位置、特徴量がない顔は追跡できません。
            guard let personID = face.personID, face.boundingBox != nil, face.embedding != nil else { continue }
            // 古すぎる顔を候補にすると別人へつながるので除きます。
            guard input.video.lastAnalyzedSecond - face.second <= temporalGap + 0.25 else { continue }
            // 同じ人物では最も新しい顔だけを残します。
            if face.second > (recentByPerson[personID]?.second ?? -.infinity) { recentByPerson[personID] = face }
        // ここで直前の処理の範囲を閉じます。
        }
        // 同時に映った顔の位置を一覧にして追跡の競合を検出します。
        var indicesByFrame: [Int: [Int]] = [:]
        // 各検出顔の実フレーム時刻と配列位置を記録します。
        for (index, detected) in input.result.faces.enumerated() { indicesByFrame[Int((detected.second * 600).rounded()), default: []].append(index) }
        // 今回の顔を保存する配列です。
        var newFaces: [FaceRecord] = []
        // 今回できた人物を保存する配列です。
        var newPeople: [PersonRecord] = []
        // 同じ旧顔を複数回引き継がないために使います。
        var matched = Set<UUID>()
        // 今回検出した顔を順に分類します。
        for (detectedIndex, detected) in input.result.faces.enumerated() {
            // 実際に読まれたフレーム時刻を六百分の一秒単位にします。
            let frameKey = Int((detected.second * 600).rounded())
            // 同時刻の未使用の旧顔を位置や特徴量で比較します。
            let candidates = (previousByFrame[frameKey] ?? []).filter { !matched.contains($0.id) }.compactMap { old -> (FaceRecord, Double)? in
                // 一致度を得られない旧顔は候補から外します。
                guard let score = faceMatchScore(old, detected) else { return nil }
                // 旧顔と一致度を候補として返します。
                return (old, score)
            // ここで直前の処理の範囲を閉じます。
            }
            // 最もよく一致する旧顔の手動設定を引き継ぎます。
            let reusable = candidates.min { $0.1 < $1.1 }?.0
            // 旧顔を再利用する場合は ID を保持します。
            let id = reusable?.id ?? UUID()
            // 二つ目の新顔が同じ旧顔を使わないようにします。
            if let reusable { matched.insert(reusable.id) }
            // 顔画像をアプリ専用の画像保存先へ置きます。
            let imageURL = input.thumbnailDirectory.appendingPathComponent("face-\(id.uuidString).jpg")
            // 検出画像をディスクに保存します。
            try detected.thumbnailData.write(to: imageURL, options: .atomic)
            // 同じ場所に続けて映る顔だけから慎重な追跡候補を選びます。
            let temporalID = reusable == nil ? temporalPerson(for: detected, at: detectedIndex, peers: indicesByFrame[frameKey] ?? [], detections: input.result.faces, recentByPerson: recentByPerson, occupiedPeople: occupiedAtFrame[frameKey, default: []], maximumGap: temporalGap) : nil
            // 旧顔がなければ特徴量で近い人物を探します。
            let personID: UUID?
            // 手動編集済みの旧顔なら割当を維持します。
            if let reusable { personID = reusable.personID }
            // 新しい顔だけ既存人物との距離を比べます。
            else { personID = clusterPerson(for: detected.embedding, thumbnailPath: imageURL.path, representatives: representatives, occupiedPeople: occupiedAtFrame[frameKey, default: []], temporalID: temporalID, newPeople: &newPeople) }
            // 画像、時刻、割当を一件の顔情報にまとめます。
            let face = FaceRecord(id: id, videoID: videoID, second: detected.second, boundingBox: detected.boundingBox, personID: personID, thumbnailPath: imageURL.path, embedding: detected.embedding, manualAssignment: reusable?.manualAssignment ?? false, excluded: reusable?.excluded ?? false)
            // 後でまとめて保存できるよう配列に追加します。
            newFaces.append(face)
            // 除外されていない顔だけ同時出現と代表顔に反映します。
            if let personID, !face.excluded {
                // 同じフレームで別の顔がこの人物を選ばないようにします。
                occupiedAtFrame[frameKey, default: []].insert(personID)
                // 新人物も後続のフレームから照合できるようにします。
                addRepresentative(face, to: &representatives)
                // 次の時刻の顔からはこの新しい顔を直近の追跡起点にします。
                recentByPerson[personID] = face
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 動画情報は呼び出し元と共有しない値の写しとして更新します。
        var updated = input.video
        // 顔が見つからない区間でも動画の長さを記録します。
        updated.duration = input.result.duration
        // 区間の終端を次回の再開位置にします。
        updated.lastAnalyzedSecond = end
        // 最後まで読んだときだけ解析完了とします。
        updated.state = input.result.wasCancelled || end < input.result.duration ? .pending : .ready
        // 成功した区間では以前のエラーを消します。
        updated.errorMessage = nil
        // 最初の区間で必要な場合だけポスターを書きます。
        if updated.posterPath == nil, let image = input.result.posterData {
            // 動画 ID からポスター画像の保存先を作ります。
            let url = input.thumbnailDirectory.appendingPathComponent("poster-\(videoID.uuidString).jpg")
            // 一覧表示用の代表画像を保存します。
            try image.write(to: url, options: .atomic)
            // 保存できた画像を動画情報から参照します。
            updated.posterPath = url.path
        // ここで直前の処理の範囲を閉じます。
        }
        // 画面の SQLite 接続と分けた背景用接続を開きます。
        let database = try LibraryDatabase(databaseURL: input.databaseURL)
        // 顔、人物、再開位置を一つのトランザクションで保存します。
        try database.applyFaceBatch(video: updated, newPeople: newPeople, faces: newFaces, removing: oldIDs)
        // 旧区間の顔と新しい顔だけを画面へ返します。
        return FaceBatchOutput(video: updated, faces: newFaces, people: newPeople, removedFaceIDs: oldIDs)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同じ人物の似た顔を省き、異なる顔向きの代表を八枚まで保持します。
    static func addRepresentative(_ face: FaceRecord, to representatives: inout [UUID: [FaceRecord]]) {
        // 分類済みで比較できる顔だけを代表に使います。
        guard !face.excluded, let personID = face.personID, face.embedding != nil else { return }
        // 現在この人物に保存済みの代表を値として取得します。
        var examples = representatives[personID] ?? []
        // まだ代表がなければ最初の顔を基準として固定します。
        if examples.isEmpty { representatives[personID] = [face]; return }
        // 既に同じ顔 ID を持つ代表は重複させません。
        if examples.contains(where: { $0.id == face.id }) { return }
        // 新しい顔から既存代表への最短距離を求めます。
        let novelty = examples.reduce(Double.infinity) { min($0, faceDistance($1.embedding, face.embedding)) }
        // ほぼ同じ姿勢の顔は比較件数を増やさずに済ませます。
        guard novelty >= 0.10 else { return }
        // 八枚未満なら異なる顔をそのまま追加します。
        if examples.count < 8 { examples.append(face); representatives[personID] = examples; return }
        // 最初の基準顔を残し、残りから最も重複した代表を探します。
        var redundantIndex = 1
        // 他の代表への近さが最小の候補を記録します。
        var smallestSeparation = Double.infinity
        // 基準顔以外の各代表を調べます。
        for index in 1..<examples.count {
            // この代表から他の代表への最短距離を求めます。
            var separation = Double.infinity
            // 自身以外の代表との距離を比較します。
            for other in examples.indices where other != index { separation = min(separation, faceDistance(examples[index].embedding, examples[other].embedding)) }
            // より重複した代表が見つかれば入替候補にします。
            if separation < smallestSeparation { smallestSeparation = separation; redundantIndex = index }
        // ここで直前の処理の範囲を閉じます。
        }
        // 新顔が既存の重複より十分に異なる場合だけ入れ替えます。
        if novelty > smallestSeparation + 0.03 { examples[redundantIndex] = face; representatives[personID] = examples }
    // ここで直前の処理の範囲を閉じます。
    }

    // 顔特徴量間のコサイン距離を計算します。
    private static func faceDistance(_ left: [Float]?, _ right: [Float]?) -> Double {
        // 欠損や次元違いの特徴量は比較不能とします。
        guard let left, let right, !left.isEmpty, left.count == right.count else { return .infinity }
        // 内積を倍精度で蓄積します。
        var dot = 0.0
        // 左側の長さの二乗です。
        var leftLength = 0.0
        // 右側の長さの二乗です。
        var rightLength = 0.0
        // 全次元の積と長さを足します。
        for index in left.indices {
            // 左側の値を倍精度へ変えます。
            let a = Double(left[index])
            // 右側の値を倍精度へ変えます。
            let b = Double(right[index])
            // 内積へ加算します。
            dot += a * b
            // 左側の長さへ加算します。
            leftLength += a * a
            // 右側の長さへ加算します。
            rightLength += b * b
        // ここで直前の処理の範囲を閉じます。
        }
        // 長さがゼロなら比較不能とします。
        guard leftLength > 0, rightLength > 0 else { return .infinity }
        // 1 からコサイン類似度を引いて距離にします。
        return 1 - dot / sqrt(leftLength * rightLength)
    // ここで直前の処理の範囲を閉じます。
    }

    // 再解析した顔を旧顔へ対応付けるスコアです。
    private static func faceMatchScore(_ old: FaceRecord, _ detected: AnalyzedFace) -> Double? {
        // 旧顔の位置があれば顔枠の重なりを比べます。
        if let oldBox = old.boundingBox {
            // 二つの顔枠が重なる領域を計算します。
            let overlap = oldBox.intersection(detected.boundingBox)
            // 重ならない領域の面積はゼロとします。
            let area = overlap.isNull ? 0 : overlap.width * overlap.height
            // 二つの顔枠を合わせた面積を求めます。
            let total = oldBox.width * oldBox.height + detected.boundingBox.width * detected.boundingBox.height - area
            // 顔枠が十分重ならなければ別の検出結果です。
            guard total > 0, area / total >= 0.3 else { return nil }
            // 顔枠の重なりが大きいほど低いスコアを返します。
            return 1 - Double(area / total)
        // ここで直前の処理の範囲を閉じます。
        }
        // 旧形式で顔枠がないときだけ特徴量を比べます。
        let distance = faceDistance(old.embedding, detected.embedding)
        // 十分近い旧顔だけを再利用します。
        return distance <= 0.25 ? 1 + distance : nil
    // ここで直前の処理の範囲を閉じます。
    }

    // 前後フレームの顔が同じ人物として続く可能性を数値で表します。
    private static func temporalScore(from prior: FaceRecord, to detected: AnalyzedFace, maximumGap: Double) -> Double? {
        // 顔が時間的に前後し、許容する短い間隔に収まるか確かめます。
        let gap = detected.second - prior.second
        // 同一時刻や長い不在をつなぐと別人が混ざるため除きます。
        guard gap > 0, gap <= maximumGap else { return nil }
        // 顔枠と特徴量のどちらかを欠く場合は追跡しません。
        guard let oldBox = prior.boundingBox, prior.embedding != nil, detected.embedding != nil else { return nil }
        // 二つの顔枠が重なる部分を求めます。
        let overlap = oldBox.intersection(detected.boundingBox)
        // 重なりがなければ面積をゼロにします。
        let area = overlap.isNull ? 0 : overlap.width * overlap.height
        // 両方の顔枠を合わせた面積を求めます。
        let total = oldBox.width * oldBox.height + detected.boundingBox.width * detected.boundingBox.height - area
        // 顔枠の半分以上が重なるときだけ同じ位置とみなします。
        guard total > 0, area / total >= 0.5 else { return nil }
        // 顔特徴量が大きく変わりすぎていないか調べます。
        let distance = faceDistance(prior.embedding, detected.embedding)
        // 照合距離が 0.45 を超える場合は位置だけでつなぎません。
        guard distance <= 0.45 else { return nil }
        // 特徴量、顔枠、時間差を使い、値が低いほど強い候補にします。
        return distance + 0.2 * (1 - Double(area / total)) + 0.01 * gap / maximumGap
    // ここで直前の処理の範囲を閉じます。
    }

    // 同じ時刻の別の顔や別の人物と競合しない追跡先だけを返します。
    private static func temporalPerson(for detected: AnalyzedFace, at index: Int, peers: [Int], detections: [AnalyzedFace], recentByPerson: [UUID: FaceRecord], occupiedPeople: Set<UUID>, maximumGap: Double) -> UUID? {
        // 比較できる顔だけを時間的に追跡します。
        guard detected.embedding != nil else { return nil }
        // 各人物の最新顔から有効な候補を作ります。
        var candidates: [(id: UUID, score: Double)] = []
        // 同時刻に使われた人物を除いて追跡を試します。
        for (personID, prior) in recentByPerson where !occupiedPeople.contains(personID) {
            // 時間、位置、特徴量の条件を満たす候補だけを残します。
            if let score = temporalScore(from: prior, to: detected, maximumGap: maximumGap) { candidates.append((personID, score)) }
        // ここで直前の処理の範囲を閉じます。
        }
        // より確かな候補から順番に並べます。
        candidates.sort { $0.score < $1.score }
        // 候補がなければ追跡による割当をしません。
        guard let best = candidates.first, let prior = recentByPerson[best.id] else { return nil }
        // 別人物の候補がほぼ同じ強さなら誤統合を避けます。
        if candidates.count > 1 && candidates[1].score - best.score < 0.05 { return nil }
        // 同時に映った別の顔も同じ過去の人物を求めるか調べます。
        for peerIndex in peers where peerIndex != index {
            // 別の顔にも同じ強さの対応があれば一意ではありません。
            if let score = temporalScore(from: prior, to: detections[peerIndex], maximumGap: maximumGap), score <= best.score + 0.05 { return nil }
        // ここで直前の処理の範囲を閉じます。
        }
        // 同時刻の顔と競合しない人物 ID を返します。
        return best.id
    // ここで直前の処理の範囲を閉じます。
    }

    // 近い人物がいれば選び、なければ顔写真のみの新人物を作ります。
    private static func clusterPerson(for embedding: [Float]?, thumbnailPath: String, representatives: [UUID: [FaceRecord]], occupiedPeople: Set<UUID>, temporalID: UUID?, newPeople: inout [PersonRecord]) -> UUID? {
        // 顔を比較できないときは未分類に残します。
        guard embedding != nil else { return nil }
        // しきい値を通った人物候補を保持します。
        var candidates: [(id: UUID, distance: Double)] = []
        // 同時に写った人物を除いて既存の代表顔を比べます。
        for (personID, examples) in representatives where !occupiedPeople.contains(personID) {
            // 同じ人物の最大三枚から最短距離を求めます。
            let distance = examples.reduce(Double.infinity) { min($0, faceDistance($1.embedding, embedding)) }
            // 慎重なしきい値以内の候補だけ残します。
            if distance <= 0.25 { candidates.append((personID, distance)) }
        // ここで直前の処理の範囲を閉じます。
        }
        // 最も近い人物を先頭に並べます。
        candidates.sort { $0.distance < $1.distance }
        // 近さが拮抗する場合は誤統合を避けます。
        if candidates.count > 1 && candidates[1].distance - candidates[0].distance < 0.05 { return nil }
        // 明確に近い人物がいればその ID を返します。
        if let best = candidates.first {
            // 強い全動画照合と短時間追跡が違う人物を示したら未分類にします。
            guard temporalID == nil || temporalID == best.id else { return nil }
            // 両方の根拠が矛盾しない人物を選びます。
            return best.id
        // ここで直前の処理の範囲を閉じます。
        }
        // 全動画照合が見つからない場合は一意な短時間追跡を採用します。
        if let temporalID { return temporalID }
        // 表示名を付けずに顔写真だけを持つ人物を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: thumbnailPath)
        // 呼び出し元が顔より先に人物を保存できるようにします。
        newPeople.append(person)
        // 新しい人物の ID を返します。
        return person.id
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// 解析済みライブラリの顔と動画を背景の統合処理へ渡します。
struct GroupReconciliationInput: @unchecked Sendable {
    // 手動修正や同時出現も含む顔の写しです。
    let faces: [FaceRecord]
    // 有効な動画だけを判別するための写しです。
    let videos: [VideoRecord]
    // 存在する人物 ID を判別するための写しです。
    let people: [PersonRecord]
    // 統合した分類を書き込むデータベースの場所です。
    let databaseURL: URL
// ここで直前の処理の範囲を閉じます。
}

// 自動統合で変更した人物 ID と件数を画面へ返します。
struct GroupReconciliationOutput: Sendable {
    // 統合元から最終的な統合先への対応です。
    let personMapping: [UUID: UUID]
    // SQLite に更新した顔の件数です。
    let changedFaceCount: Int
    // 詳細比較まで進めたグループの組数です。
    let comparedPairCount: Int
// ここで直前の処理の範囲を閉じます。
}

// 複数の顔で十分に裏付けられた人物グループだけを自動で統合します。
enum GroupReconciler {
    // 二つの元グループの順序に依存しない検索キーです。
    private struct GroupPair: Hashable {
        // 辞書順で前の人物 ID です。
        let first: UUID
        // 辞書順で後の人物 ID です。
        let second: UUID
        // 呼び出し順にかかわらず同じ組を一つのキーにします。
        init(_ left: UUID, _ right: UUID) {
            // 一つ目の ID を順序付けます。
            first = left.uuidString < right.uuidString ? left : right
            // 二つ目の ID を順序付けます。
            second = left.uuidString < right.uuidString ? right : left
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }
    // 時間的に同じフレームに現れた二つのグループを分離するキーです。
    private struct FrameReference: Hashable {
        // 顔が映った動画の ID です。
        let videoID: UUID
        // 実フレーム時刻を十分の一秒単位に丸めた値です。
        let tenthSecond: Int
    // ここで直前の処理の範囲を閉じます。
    }
    // 顔特徴量を長さ一へ正規化して保持します。
    private struct Sample {
        // コサイン距離を速く計算できる単位ベクトルです。
        let unit: [Float]
    // ここで直前の処理の範囲を閉じます。
    }
    // 一つの人物グループの比較情報です。
    private struct Group {
        // 元の人物 ID です。
        let id: UUID
        // 異なる時刻から最大三十二枚の顔を選びます。
        let samples: [Sample]
        // 最終的な統合条件にはこのグループの全顔を使います。
        let allSamples: [Sample]
        // 粗い候補絞りに使う平均ベクトルです。
        let center: [Float]
        // 平均から最も遠い代表顔までの距離です。
        let radius: Float
        // 顔が映った動画の集合です。
        let videoIDs: Set<UUID>
        // 同じフレームで別人が映ったか判別する集合です。
        let frames: Set<FrameReference>
        // 大きいグループを統合先として残すための顔の件数です。
        let faceCount: Int
    // ここで直前の処理の範囲を閉じます。
    }

    // 顔特徴量を長さ一にそろえてコサイン距離を計算しやすくします。
    private static func normalizedSample(_ face: FaceRecord, dimension: Int) -> Sample? {
        // 指定した特徴量モデルの次元だけを使います。
        guard let embedding = face.embedding, embedding.count == dimension else { return nil }
        // ベクトルの長さの二乗を計算します。
        let lengthSquared = embedding.reduce(Float(0)) { $0 + $1 * $1 }
        // 空や不正な値では人物を比較しません。
        guard lengthSquared.isFinite, lengthSquared > 0 else { return nil }
        // 各値を同じ長さへ正規化して返します。
        return Sample(unit: embedding.map { $0 / sqrt(lengthSquared) })
    // ここで直前の処理の範囲を閉じます。
    }
    // 二つのグループを比較した支持量です。
    private struct Evidence {
        // 一つ目の人物 ID です。
        let first: UUID
        // 二つ目の人物 ID です。
        let second: UUID
        // 最も近い顔のコサイン距離です。
        let minimumDistance: Double
        // 一つ目から相手へ 0.25 以内で近い顔の件数です。
        let firstSupport: Int
        // 二つ目から相手へ 0.25 以内で近い顔の件数です。
        let secondSupport: Int
        // 一つ目の代表顔のうち支持した割合です。
        let firstFraction: Double
        // 二つ目の代表顔のうち支持した割合です。
        let secondFraction: Double
        // 同じ動画にも映る候補はより厳しい条件を適用します。
        func accepts(sharedVideo: Bool) -> Bool {
            // 同じ動画で別人を混ぜないよう最小件数を増やします。
            let minimumCount = sharedVideo ? 10 : 5
            // 同じ動画では代表顔の四分の一以上を要求します。
            let minimumFraction = sharedVideo ? 0.25 : 0.10
            // 同じ動画では一段近い顔の一致も要求します。
            let strongDistance = sharedVideo ? 0.18 : 0.20
            // 双方向の複数時刻で支持される組だけを通します。
            return firstSupport >= minimumCount && secondSupport >= minimumCount && firstFraction >= minimumFraction && secondFraction >= minimumFraction && minimumDistance <= strongDistance
        // ここで直前の処理の範囲を閉じます。
        }
        // 支持が広く距離が近い組を先に統合する値です。
        var priority: Double {
            // 弱い側の支持率を重視し、近い顔に小さい加点を与えます。
            min(firstFraction, secondFraction) + 0.1 * (1 - minimumDistance)
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }

    // 全解析済み動画を一度だけ比較し、安全な人物 ID の変更を保存します。
    static func reconcile(_ input: GroupReconciliationInput) throws -> GroupReconciliationOutput {
        // 準備済み動画の顔だけを比較します。
        let readyVideoIDs = Set(input.videos.filter { $0.state == .ready }.map(\.id))
        // 手動で編集した人物グループは自動統合の対象から外します。
        let protectedIDs = Set(input.faces.filter { $0.manualAssignment }.compactMap(\.personID))
        // SQLite に存在する人物だけを候補にします。
        let existingIDs = Set(input.people.map(\.id))
        // 除外されておらず特徴量のある顔を人物ごとに集めます。
        let grouped = Dictionary(grouping: input.faces.filter { !$0.excluded && readyVideoIDs.contains($0.videoID) && $0.embedding != nil && $0.personID != nil && !protectedIDs.contains($0.personID!) && existingIDs.contains($0.personID!) }, by: { $0.personID! })
        // 顔が十分にある人物だけから比較用の情報を作ります。
        let groups = grouped.compactMap { identifier, records -> Group? in
            // 顔を時間と動画に分散させて最大三十二枚選びます。
            let selected = balancedFaces(records, limit: 32)
            // 双方向で五枚を要求するので少なすぎる人物を外します。
            guard selected.count >= 5 else { return nil }
            // 特徴量の次元を最初の顔に合わせます。
            let dimension = selected[0].embedding?.count ?? 0
            // 空や不揃いの特徴量は比較しません。
            guard dimension > 0, selected.allSatisfy({ $0.embedding?.count == dimension }) else { return nil }
            // 少数の顔を候補の絞り込み用に正規化します。
            let samples = selected.compactMap { normalizedSample($0, dimension: dimension) }
            // 不正な値が多いグループは統合しません。
            guard samples.count >= 5 else { return nil }
            // 最終的な根拠は全時刻の顔から計算します。
            let allSamples = records.compactMap { normalizedSample($0, dimension: dimension) }
            // 各次元の平均をゼロから作ります。
            var center = [Float](repeating: 0, count: dimension)
            // 全顔を含む平均を作って粗い除外判定を安全にします。
            for sample in allSamples {
                // 全次元の値を足します。
                for index in center.indices { center[index] += sample.unit[index] / Float(allSamples.count) }
            // ここで直前の処理の範囲を閉じます。
            }
            // 平均から最も離れた代表までの距離を求めます。
            let radius = allSamples.map { sample -> Float in
                // 各次元の差の二乗を合計します。
                let squared = center.indices.reduce(Float(0)) { $0 + (sample.unit[$1] - center[$1]) * (sample.unit[$1] - center[$1]) }
                // ユークリッド距離として返します。
                return sqrt(squared)
            // ここで直前の処理の範囲を閉じます。
            }.max() ?? 0
            // 同じ動画に映った候補を判別します。
            let videoIDs = Set(records.map(\.videoID))
            // 同時に映った人物を混ぜないための時刻集合です。
            let frames = Set(records.map { FrameReference(videoID: $0.videoID, tenthSecond: Int(($0.second * 10).rounded())) })
            // 統合先の選択に使う件数も含めて返します。
            return Group(id: identifier, samples: samples, allSamples: allSamples, center: center, radius: radius, videoIDs: videoIDs, frames: frames, faceCount: records.count)
        // ここで直前の処理の範囲を閉じます。
        }
        // 比較対象が二人未満なら変更しません。
        guard groups.count >= 2 else { return GroupReconciliationOutput(personMapping: [:], changedFaceCount: 0, comparedPairCount: 0) }
        // 十分に近い顔が複数ある組を保持します。
        var edges: [Evidence] = []
        // 統合後にも元グループ間の全組に根拠があるか調べる辞書です。
        var evidenceByPair: [GroupPair: Evidence] = [:]
        // 詳細比較まで進んだ組数を数えます。
        var comparedPairs = 0
        // 重複する組を調べないよう添字を増やします。
        for firstIndex in groups.indices {
            // 二人目は一人目より後だけを選びます。
            for secondIndex in (firstIndex + 1)..<groups.count {
                // 二つの候補を値として取り出します。
                let first = groups[firstIndex]
                // 二つ目の候補を取り出します。
                let second = groups[secondIndex]
                // 同じフレームに二人が映る場合は統合しません。
                guard first.frames.isDisjoint(with: second.frames) else { continue }
                // 特徴量の次元が異なるモデルの結果は比較しません。
                guard first.center.count == second.center.count else { continue }
                // 平均と半径から近い顔があり得ない組を省きます。
                guard couldContainCloseFaces(first, second) else { continue }
                // 少数の顔にも 0.35 以内の手掛かりがない組は全件比較しません。
                guard hasCoarseCandidate(first, second) else { continue }
                // 詳細比較した組を記録します。
                comparedPairs += 1
                // 候補については全顔から双方向の支持量を得ます。
                guard let evidence = compare(first, second) else { continue }
                // 同じ動画に現れる場合は厳しい条件を使います。
                let sharedVideo = !first.videoIDs.isDisjoint(with: second.videoIDs)
                // 条件を満たす組だけ自動統合候補にします。
                if evidence.accepts(sharedVideo: sharedVideo) {
                    // 強い候補を優先順位付きの一覧へ加えます。
                    edges.append(evidence)
                    // 後で統合集合の全組を確かめられるよう保存します。
                    evidenceByPair[GroupPair(first.id, second.id)] = evidence
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 動画をまたぐ確かな同一人物を先に結び、同じ動画の断片は後で扱います。
        let originalVideos = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0.videoIDs) })
        // 強い跨動画の根拠を同一動画内の断片統合より優先します。
        edges.sort { left, right in
            // 一つ目の辺が別動画同士か調べます。
            let leftCrossVideo = originalVideos[left.first]!.isDisjoint(with: originalVideos[left.second]!)
            // 二つ目の辺が別動画同士か調べます。
            let rightCrossVideo = originalVideos[right.first]!.isDisjoint(with: originalVideos[right.second]!)
            // 種類が違えば跨動画の辺を先に処理します。
            if leftCrossVideo != rightCrossVideo { return leftCrossVideo }
            // 同じ種類の辺は弱い側の支持率が高い順に処理します。
            return left.priority > right.priority
        // ここで直前の処理の範囲を閉じます。
        }
        // 各人物を最初は別々の統合集合として保持します。
        var parent = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0.id) })
        // 統合集合に現れる動画を保持します。
        var componentVideos = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0.videoIDs) })
        // 統合集合に現れるフレームを保持します。
        var componentFrames = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0.frames) })
        // 統合先には顔が多い側の人物を残します。
        var componentSizes = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0.faceCount) })
        // それぞれの統合集合に含まれる元グループを保持します。
        var componentMembers = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, Set([$0.id])) })
        // 強い根拠を持つ各組を一度だけ統合します。
        for edge in edges {
            // 一つ目の現在の統合先を探します。
            let firstRoot = root(of: edge.first, parent: parent)
            // 二つ目の現在の統合先を探します。
            let secondRoot = root(of: edge.second, parent: parent)
            // 既に一つの人物なら再統合しません。
            guard firstRoot != secondRoot else { continue }
            // 統合済みの他グループを含めても同時出現しないことを確認します。
            guard componentFrames[firstRoot]!.isDisjoint(with: componentFrames[secondRoot]!) else { continue }
            // 統合済みの動画集合に共通動画があれば厳しい支持量を要求します。
            let sharedVideo = !componentVideos[firstRoot]!.isDisjoint(with: componentVideos[secondRoot]!)
            // 元の組の根拠が現在の条件にも通る場合だけ統合します。
            guard edge.accepts(sharedVideo: sharedVideo) else { continue }
            // 橋渡し一組だけで無関係な二人を連鎖統合しないようにします。
            let everyPairSupported = componentMembers[firstRoot]!.allSatisfy { firstID in
                // もう一方の集合のすべての元グループを調べます。
                componentMembers[secondRoot]!.allSatisfy { secondID in
                    // 元グループ間に強い双方向支持がなければ連結しません。
                    evidenceByPair[GroupPair(firstID, secondID)]?.accepts(sharedVideo: sharedVideo) == true
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
            // 全組に根拠がない曖昧な橋渡しは飛ばします。
            guard everyPairSupported else { continue }
            // 顔の数が多い人物 ID を残します。
            let keep = componentSizes[firstRoot]! >= componentSizes[secondRoot]! ? firstRoot : secondRoot
            // もう一方の人物 ID は統合元にします。
            let remove = keep == firstRoot ? secondRoot : firstRoot
            // 統合元から最終人物への親を記録します。
            parent[remove] = keep
            // 動画集合を統合先へまとめます。
            componentVideos[keep]!.formUnion(componentVideos[remove]!)
            // 同時出現の照合用時刻を統合先へまとめます。
            componentFrames[keep]!.formUnion(componentFrames[remove]!)
            // 統合先の顔件数を更新します。
            componentSizes[keep]! += componentSizes[remove]!
            // 統合済みの元グループ一覧もまとめます。
            componentMembers[keep]!.formUnion(componentMembers[remove]!)
        // ここで直前の処理の範囲を閉じます。
        }
        // 統合元を最終的な人物 ID へ対応付けます。
        var mapping: [UUID: UUID] = [:]
        // 元の各人物について最終 ID を調べます。
        for group in groups {
            // 親をたどって最終統合先を得ます。
            let destination = root(of: group.id, parent: parent)
            // 自分以外へ移動する人物だけを保存対象にします。
            if destination != group.id { mapping[group.id] = destination }
        // ここで直前の処理の範囲を閉じます。
        }
        // 根拠が足りなければ SQLite を書き換えません。
        guard !mapping.isEmpty else { return GroupReconciliationOutput(personMapping: [:], changedFaceCount: 0, comparedPairCount: comparedPairs) }
        // 統合元に割り当てられた顔だけを保存用に複製します。
        let changedFaces = input.faces.compactMap { face -> FaceRecord? in
            // 統合元でない顔は変更しません。
            guard let sourceID = face.personID, let destination = mapping[sourceID] else { return nil }
            // 手動修正が混ざった場合は安全のため統合を失敗させます。
            guard !face.manualAssignment else { return nil }
            // 元の顔の位置、画像、除外状態を保ったまま人物 ID だけ変えます。
            var changed = face
            // 新しい人物 ID を割り当てます。
            changed.personID = destination
            // 一件の更新として返します。
            return changed
        // ここで直前の処理の範囲を閉じます。
        }
        // 自動統合だけを別の SQLite 接続で一括確定します。
        let database = try LibraryDatabase(databaseURL: input.databaseURL)
        // 変更した顔と統合元の削除を同じトランザクションで保存します。
        try database.applyAutomaticPersonMerges(updatedFaces: changedFaces, removing: Set(mapping.keys))
        // 完了した変更数を画面へ返します。
        return GroupReconciliationOutput(personMapping: mapping, changedFaceCount: changedFaces.count, comparedPairCount: comparedPairs)
    // ここで直前の処理の範囲を閉じます。
    }

    // 大きな動画内でも各動画の複数時刻から代表顔を選びます。
    private static func balancedFaces(_ faces: [FaceRecord], limit: Int) -> [FaceRecord] {
        // 同じ動画の顔を時間順にまとめます。
        let byVideo = Dictionary(grouping: faces, by: \.videoID)
        // 動画 ID の順序を固定して再実行結果を安定させます。
        let videoIDs = byVideo.keys.sorted { $0.uuidString < $1.uuidString }
        // 一動画から最低一件かつ均等に取る数を求めます。
        let quota = max(1, limit / max(1, videoIDs.count))
        // 選んだ顔を一時的に入れます。
        var selected: [FaceRecord] = []
        // 各動画の序盤と後半を広く取ります。
        for videoID in videoIDs {
            // この動画の顔を時刻順に並べます。
            let ordered = (byVideo[videoID] ?? []).sorted { $0.second < $1.second }
            // 一動画から取る実際の枚数です。
            let count = min(quota, ordered.count)
            // 均等な間隔で顔を選びます。
            selected.append(contentsOf: evenlySpaced(ordered, count: count))
        // ここで直前の処理の範囲を閉じます。
        }
        // 同じ顔を二度選ばないため ID を集めます。
        let chosenIDs = Set(selected.map(\.id))
        // 空いた枠にはまだ使っていない時刻の顔を選びます。
        let remainder = faces.filter { !chosenIDs.contains($0.id) }.sorted { $0.second < $1.second }
        // 残りの枠へ時間的に分散した顔を追加します。
        selected.append(contentsOf: evenlySpaced(remainder, count: max(0, limit - selected.count)))
        // 一組あたり最大三十二枚だけ返します。
        return Array(selected.prefix(limit))
    // ここで直前の処理の範囲を閉じます。
    }

    // 時間順の顔から指定枚数を均等に選びます。
    private static func evenlySpaced(_ faces: [FaceRecord], count: Int) -> [FaceRecord] {
        // 顔がなければ何も選びません。
        guard !faces.isEmpty, count > 0 else { return [] }
        // 要求枚数が全件以上ならすべて返します。
        guard count < faces.count else { return faces }
        // 一枚だけなら中央付近を代表にします。
        if count == 1 { return [faces[faces.count / 2]] }
        // 最初から最後まで等間隔の添字を選びます。
        return (0..<count).map { index in faces[Int((Double(index) * Double(faces.count - 1) / Double(count - 1)).rounded())] }
    // ここで直前の処理の範囲を閉じます。
    }

    // どの代表同士も 0.25 以内になれない組を安全に省きます。
    private static func couldContainCloseFaces(_ first: Group, _ second: Group) -> Bool {
        // 異なる特徴量次元の組は比較しません。
        guard first.center.count == second.center.count else { return false }
        // 平均ベクトル間のユークリッド距離を二乗で求めます。
        let squared = first.center.indices.reduce(Float(0)) { $0 + (first.center[$1] - second.center[$1]) * (first.center[$1] - second.center[$1]) }
        // 三角不等式で最も近づき得る距離を求めます。
        let lowerBound = sqrt(squared) - first.radius - second.radius
        // 単位ベクトルのコサイン距離 0.25 はユークリッド距離 √0.5 です。
        return lowerBound <= sqrt(Float(0.5))
    // ここで直前の処理の範囲を閉じます。
    }

    // 少数の代表から全顔比較へ進める候補を素早く選びます。
    private static func hasCoarseCandidate(_ first: Group, _ second: Group) -> Bool {
        // 一つ目の時間分散した顔を順番に調べます。
        for left in first.samples {
            // 二つ目の時間分散した顔と比べます。
            for right in second.samples {
                // 最終条件より緩い距離に一組あれば全件の照合へ進みます。
                if cosineDistance(left.unit, right.unit) <= 0.35 { return true }
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 粗い代表に手掛かりがない組は候補から外します。
        return false
    // ここで直前の処理の範囲を閉じます。
    }

    // 長さ一の二つの顔特徴量からコサイン距離を求めます。
    private static func cosineDistance(_ first: [Float], _ second: [Float]) -> Double {
        // 内積を単精度で蓄積します。
        var dot: Float = 0
        // 各次元を一度ずつ掛け合わせます。
        for index in first.indices { dot += first[index] * second[index] }
        // 単位ベクトルの内積を零以上の距離に変換します。
        return max(0, 1 - Double(dot))
    // ここで直前の処理の範囲を閉じます。
    }

    // 二組の全代表を比べ、互いに近い顔の件数を数えます。
    private static func compare(_ first: Group, _ second: Group) -> Evidence? {
        // 一つ目の各顔から二つ目への最短距離を記録します。
        var nearestFirst = [Double](repeating: .infinity, count: first.allSamples.count)
        // 二つ目の各顔から一つ目への最短距離を記録します。
        var nearestSecond = [Double](repeating: .infinity, count: second.allSamples.count)
        // 最も近い顔の距離を保持します。
        var minimum = Double.infinity
        // 一つ目の各顔を順番に比較します。
        for firstIndex in first.allSamples.indices {
            // 二つ目の各顔を順番に比較します。
            for secondIndex in second.allSamples.indices {
                // 全顔同士のコサイン距離を求めます。
                let distance = cosineDistance(first.allSamples[firstIndex].unit, second.allSamples[secondIndex].unit)
                // 一つ目から見た最短距離を更新します。
                nearestFirst[firstIndex] = min(nearestFirst[firstIndex], distance)
                // 二つ目から見た最短距離を更新します。
                nearestSecond[secondIndex] = min(nearestSecond[secondIndex], distance)
                // 組全体の最短距離を更新します。
                minimum = min(minimum, distance)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 一件も十分に近い顔がなければ候補から外します。
        guard minimum <= 0.20 else { return nil }
        // 一つ目から見た十分に近い顔の件数を数えます。
        let firstSupport = nearestFirst.filter { $0 <= 0.25 }.count
        // 二つ目から見た十分に近い顔の件数を数えます。
        let secondSupport = nearestSecond.filter { $0 <= 0.25 }.count
        // 双方向に五件未満なら統合できないので外します。
        guard firstSupport >= 5, secondSupport >= 5 else { return nil }
        // 双方向の件数と割合を候補として返します。
        return Evidence(first: first.id, second: second.id, minimumDistance: minimum, firstSupport: firstSupport, secondSupport: secondSupport, firstFraction: Double(firstSupport) / Double(first.allSamples.count), secondFraction: Double(secondSupport) / Double(second.allSamples.count))
    // ここで直前の処理の範囲を閉じます。
    }

    // 統合済みの親をたどって最終的な人物 ID を得ます。
    private static func root(of identifier: UUID, parent: [UUID: UUID]) -> UUID {
        // 最初は指定された人物から始めます。
        var current = identifier
        // 別の人物へ統合されている間だけ親をたどります。
        while let next = parent[current], next != current { current = next }
        // 最終的に残る人物 ID を返します。
        return current
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
