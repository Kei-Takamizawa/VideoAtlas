// 一区間ずつ保存する動作を XCTest で確認します。
import XCTest
// アプリ内の保存型をテストから利用します。
@testable import VideoAtlas

// 画面側の処理回数と停止状態を主実行文脈に閉じ込めます。
@MainActor private final class ScanHeartbeat {
    // 画面側が動けた回数です。
    var ticks = 0
    // 背景保存の完了状態です。
    var finished = false
// ここで直前の処理の範囲を閉じます。
}

// 長い動画で区間ごとの保存が既存の修正を壊さないか調べます。
final class PerformanceStoreTests: XCTestCase {
    // 区間の境目で実フレーム時刻が少し早くても手動修正を引き継ぎます。
    func testEarlyActualFrameAtBatchBoundaryKeepsManualIdentity() throws {
        // このテスト専用の保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // 顔画像と SQLite 用の場所を用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト終了後に生成ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // 区間ごとの保存に使う SQLite 接続を開きます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 元データを入れる接続を作ります。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // テスト用の登録フォルダを作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 百二十秒から解析を再開する動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("sample.mp4").path, name: "sample.mp4", duration: 240, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 120, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 人が選んだ人物の ID を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: "manual.jpg")
        // 要求時刻より 0.033 秒早く読まれた顔へ手動割当を作ります。
        let old = FaceRecord(id: UUID(), videoID: video.id, second: 119.967, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), personID: person.id, thumbnailPath: "manual.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 親となる登録フォルダを保存します。
        try database.saveSource(source)
        // 動画の再開位置を保存します。
        try database.saveVideo(video)
        // 手動割当先の人物を保存します。
        try database.savePerson(person)
        // 以前の解析で得た顔を保存します。
        try database.saveFace(old)
        // 再解析でも同じ実フレーム時刻にある顔を作ります。
        let detected = AnalyzedFace(second: 119.967, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), thumbnailData: Data([0xFF, 0xD8, 0xFF, 0xD9]), embedding: [1, 0])
        // 再開後の一区間の解析結果を作ります。
        let result = VideoAnalysisResult(duration: 240, posterData: nil, faces: [detected], completedSecond: 240, wasCancelled: false)
        // 旧顔を含む保存入力を作ります。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [old], representativeFaces: [old], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 境界時刻の顔を対応付けながら保存します。
        let output = try FaceBatchStore.accept(input)
        // 以前の顔 ID が再利用されたことを確認します。
        XCTAssertEqual(output.faces.first?.id, old.id)
        // 人が選んだ人物の ID が維持されたことを確認します。
        XCTAssertEqual(output.faces.first?.personID, person.id)
        // 手動修正の印も維持されたことを確認します。
        XCTAssertEqual(output.faces.first?.manualAssignment, true)
    // ここで直前の処理の範囲を閉じます。
    }

    // 多数の顔を背景で保存しても画面の実行文脈が動けることを確認します。
    @MainActor func testBackgroundBatchAllowsMainActorHeartbeatAndSameVideoResume() async throws {
        // 他のデータに触れない一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // 顔画像とデータベース用のディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テストが終わったら生成した画像も一緒に消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // 背景側と共有するデータベースの場所を作ります。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 最初の動画登録に使う接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 一時フォルダをライブラリの登録元として作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 千二百秒の動画を検証用に作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("sample.mp4").path, name: "sample.mp4", duration: 1202, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 0, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 顔の参照先となる登録元を保存します。
        try database.saveSource(source)
        // 動画の初期情報を保存します。
        try database.saveVideo(video)
        // 同一人物を表す決まった二百五十六次元の値を作ります。
        let embedding = [Float](repeating: 1, count: 256)
        // 六百時刻分の顔を作って一括保存を試します。
        let detections = (0..<600).map { index in AnalyzedFace(second: Double(index * 2), boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), thumbnailData: Data([0xFF, 0xD8, 0xFF, 0xD9]), embedding: embedding) }
        // 最初の区間が動画末尾ではない結果を作ります。
        let result = VideoAnalysisResult(duration: 1202, posterData: nil, faces: detections, completedSecond: 1200, wasCancelled: false)
        // 背景処理へ値の写しを渡します。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [], representativeFaces: [], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 背景の保存中に画面処理が何度動いたかを数えます。
        let heartbeatState = ScanHeartbeat()
        // 多数の画像と SQLite 行を画面の実行文脈から切り離して保存します。
        let worker = Task.detached(priority: .utility) { try FaceBatchStore.accept(input) }
        // 背景保存の間も画面の実行文脈が処理できることを観察します。
        let heartbeat = Task { @MainActor in
            // 背景保存が完了するまで画面側の処理を繰り返します。
            while !heartbeatState.finished {
                // 動けた回数を記録します。
                heartbeatState.ticks += 1
                // 背景保存へ実行機会を渡します。
                await Task.yield()
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // すべての顔の保存完了を待ちます。
        let first = try await worker.value
        // 画面側の観測処理を止めます。
        heartbeatState.finished = true
        // 観測処理の終了を待ちます。
        await heartbeat.value
        // 背景保存中も画面の実行文脈が動いたことを確認します。
        XCTAssertGreaterThan(heartbeatState.ticks, 0)
        // 全六百件の顔が保存されたことを確認します。
        XCTAssertEqual(first.faces.count, 600)
        // 一人の人物へまとめられたことを確認します。
        XCTAssertEqual(Set(first.faces.compactMap(\.personID)).count, 1)
        // 再開後の同じ動画に現れる顔を作ります。
        let laterFace = AnalyzedFace(second: 1200, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), thumbnailData: Data([0xFF, 0xD8, 0xFF, 0xD9]), embedding: embedding)
        // 動画末尾までの最後の区間を作ります。
        let laterResult = VideoAnalysisResult(duration: 1202, posterData: nil, faces: [laterFace], completedSecond: 1202, wasCancelled: false)
        // 以前の同じ動画の顔を代表として再開時の入力を作ります。
        let laterInput = FaceBatchInput(result: laterResult, video: first.video, previousFaces: first.faces, representativeFaces: Array(first.faces.prefix(3)), databaseURL: databaseURL, thumbnailDirectory: directory)
        // 再開後の区間も背景保存します。
        let later = try await Task.detached(priority: .utility) { try FaceBatchStore.accept(laterInput) }.value
        // 再開後も同じ人物 ID が使われることを確認します。
        XCTAssertEqual(later.faces.first?.personID, first.faces.first?.personID)
        // 最終区間を終えた動画は解析済みになることを確認します。
        XCTAssertEqual(later.video.state, .ready)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同じ顔が二秒後に角度を変えても短時間の追跡で一人にまとめます。
    func testTemporalBridgeConnectsAdjacentFacesAtDistancePointThreeOne() throws {
        // 他のライブラリに触れない一時ディレクトリを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite と顔画像を作れるようにします。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に生成物を取り除きます。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト専用のデータベースを開きます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 外部キーの親を先に保存する接続です。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 登録元フォルダを作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 四秒の動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("a.mp4").path, name: "a.mp4", duration: 4, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 0, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 登録元を保存します。
        try database.saveSource(source)
        // 元動画を保存します。
        try database.saveVideo(video)
        // 同じ人物の顔枠を固定します。
        let box = CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3)
        // 最初の顔の特徴量を作ります。
        let first = AnalyzedFace(second: 0, boundingBox: box, thumbnailData: Data([1]), embedding: [1, 0])
        // コサイン距離が約 0.31 の二秒後の顔を作ります。
        let second = AnalyzedFace(second: 2, boundingBox: box, thumbnailData: Data([2]), embedding: [0.69, 0.724])
        // 連続する二つの顔を一区間の解析結果へまとめます。
        let result = VideoAnalysisResult(duration: 4, posterData: nil, faces: [first, second], completedSecond: 4, wasCancelled: false)
        // 初回解析に以前の顔はありません。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [], representativeFaces: [], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 特徴量と短時間追跡で人物を割り当てます。
        let output = try FaceBatchStore.accept(input)
        // 通常の 0.25 を超えても短時間の同じ位置の顔は同一人物になります。
        XCTAssertEqual(output.faces[0].personID, output.faces[1].personID)
        // 未分類ではなく一人の顔写真が作られます。
        XCTAssertEqual(output.people.count, 1)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同時に映る二人の顔を時間的に混ぜないことを確かめます。
    func testTemporalBridgeKeepsSimultaneousPeopleSeparate() throws {
        // この検証だけのディレクトリを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // データベースと顔画像を保存できるようにします。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に生成物を消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // SQLite ファイルの場所を指定します。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 登録元と動画を保存する接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 登録元の情報を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二人が同時に映る四秒の動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("pair.mp4").path, name: "pair.mp4", duration: 4, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 0, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 登録元を先に保存します。
        try database.saveSource(source)
        // 動画を保存します。
        try database.saveVideo(video)
        // 左側の顔枠を決めます。
        let left = CGRect(x: 0.1, y: 0.2, width: 0.2, height: 0.3)
        // 右側の顔枠を決めます。
        let right = CGRect(x: 0.7, y: 0.2, width: 0.2, height: 0.3)
        // 初めのフレームで左側の人物を記録します。
        let leftFirst = AnalyzedFace(second: 0, boundingBox: left, thumbnailData: Data([1]), embedding: [1, 0])
        // 同じフレームで右側の人物を記録します。
        let rightFirst = AnalyzedFace(second: 0, boundingBox: right, thumbnailData: Data([2]), embedding: [0, 1])
        // 二秒後も左側にいる顔を角度違いで記録します。
        let leftLater = AnalyzedFace(second: 2, boundingBox: left, thumbnailData: Data([3]), embedding: [0.69, 0.724])
        // 二秒後も右側にいる顔を角度違いで記録します。
        let rightLater = AnalyzedFace(second: 2, boundingBox: right, thumbnailData: Data([4]), embedding: [0.724, 0.69])
        // 四つの顔を二フレームの解析結果にまとめます。
        let result = VideoAnalysisResult(duration: 4, posterData: nil, faces: [leftFirst, rightFirst, leftLater, rightLater], completedSecond: 4, wasCancelled: false)
        // 以前の解析結果を持たない入力を作ります。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [], representativeFaces: [], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 同じ時刻の二人を含めて分類します。
        let output = try FaceBatchStore.accept(input)
        // 左側の人は二秒後も同じ人物になります。
        XCTAssertEqual(output.faces[0].personID, output.faces[2].personID)
        // 右側の人も二秒後に同じ人物になります。
        XCTAssertEqual(output.faces[1].personID, output.faces[3].personID)
        // 左右の二人が混ざらないことを確かめます。
        XCTAssertNotEqual(output.faces[0].personID, output.faces[1].personID)
    // ここで直前の処理の範囲を閉じます。
    }

    // 手動で選んだ人物を次の区間の追跡が引き継ぎます。
    func testTemporalBridgeRespectsManualPersonAcrossBatch() throws {
        // 検証専用の一時フォルダを選びます。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite と画像を作るディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 最後に一時ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // SQLite の保存先を作ります。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 初期データを保存する接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 動画の登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二秒から再開する動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("manual.mp4").path, name: "manual.mp4", duration: 4, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 2, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 手動で選ばれた人物を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: "manual.jpg")
        // 人が割り当て直した最初の顔を作ります。
        let old = FaceRecord(id: UUID(), videoID: video.id, second: 0, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), personID: person.id, thumbnailPath: "manual.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 登録元を先に保存します。
        try database.saveSource(source)
        // 動画の再開位置を保存します。
        try database.saveVideo(video)
        // 手動割当先の人物を保存します。
        try database.savePerson(person)
        // 手動修正済みの顔を保存します。
        try database.saveFace(old)
        // 二秒後に特徴量が約 0.31 異なる顔を作ります。
        let detected = AnalyzedFace(second: 2, boundingBox: old.boundingBox!, thumbnailData: Data([2]), embedding: [0.69, 0.724])
        // 次の区間の結果を作ります。
        let result = VideoAnalysisResult(duration: 4, posterData: nil, faces: [detected], completedSecond: 4, wasCancelled: false)
        // 以前の手動顔を再開時の比較候補にします。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [old], representativeFaces: [old], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 手動修正の後続顔を分類して保存します。
        let output = try FaceBatchStore.accept(input)
        // 後続顔が手動で選んだ人物に続くことを確認します。
        XCTAssertEqual(output.faces.first?.personID, person.id)
        // 人が直した以前の顔を読み直します。
        let persisted = try database.load().faces.first { $0.id == old.id }
        // 以前の顔の手動修正が残ることを確認します。
        XCTAssertEqual(persisted?.manualAssignment, true)
        // 人が指定した人物 ID も変更されていないことを確認します。
        XCTAssertEqual(persisted?.personID, person.id)
    // ここで直前の処理の範囲を閉じます。
    }

    // 後半に得た異なる顔向きが別動画との照合に使われます。
    func testLaterDiverseRepresentativeMatchesAnotherVideo() throws {
        // 他のライブラリに触れない保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite と画像用のディレクトリを準備します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に一時ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // SQLite データベースの場所を作ります。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 登録元と動画を書き込む接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 二本の動画を含む登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 既に解析済みの元動画を作ります。
        let earlierVideo = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("earlier.mp4").path, name: "earlier.mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 新たに解析する別の動画を作ります。
        let laterVideo = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("later.mp4").path, name: "later.mp4", duration: 2, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 0, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 写真だけで表示する人物を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: "anchor.jpg")
        // 外部キーの親である登録元を保存します。
        try database.saveSource(source)
        // 元動画を保存します。
        try database.saveVideo(earlierVideo)
        // 新しい動画を保存します。
        try database.saveVideo(laterVideo)
        // 既存人物を保存します。
        try database.savePerson(person)
        // 人物ごとの多様な顔を保持する辞書を作ります。
        var representatives: [UUID: [FaceRecord]] = [:]
        // 最初の三つの顔はほぼ同じ特徴量とします。
        for index in 0..<3 {
            // 同じ人物の序盤の顔を一枚作ります。
            let face = FaceRecord(id: UUID(), videoID: earlierVideo.id, second: Double(index * 2), boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), personID: person.id, thumbnailPath: "early.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
            // 序盤の顔を比較用の代表へ追加します。
            FaceBatchStore.addRepresentative(face, to: &representatives)
        // ここで直前の処理の範囲を閉じます。
        }
        // 後半の異なる顔向きを同じ人物として保存済みとします。
        let diverse = FaceRecord(id: UUID(), videoID: earlierVideo.id, second: 80, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), personID: person.id, thumbnailPath: "diverse.jpg", embedding: [0.69, 0.724], manualAssignment: false, excluded: false)
        // 後半の顔を代表集合へ追加します。
        FaceBatchStore.addRepresentative(diverse, to: &representatives)
        // 序盤の三つに限られず二つの異なる代表が残ることを確かめます。
        XCTAssertEqual(representatives[person.id]?.count, 2)
        // 別動画には後半の顔向きと同じ特徴量が現れます。
        let detected = AnalyzedFace(second: 0, boundingBox: diverse.boundingBox!, thumbnailData: Data([3]), embedding: [0.69, 0.724])
        // 別動画の結果を作ります。
        let result = VideoAnalysisResult(duration: 2, posterData: nil, faces: [detected], completedSecond: 2, wasCancelled: false)
        // 元動画の多様な顔を別動画照合に渡します。
        let input = FaceBatchInput(result: result, video: laterVideo, previousFaces: [], representativeFaces: representatives.values.flatMap { $0 }, databaseURL: databaseURL, thumbnailDirectory: directory)
        // 別動画の顔を分類します。
        let output = try FaceBatchStore.accept(input)
        // 同じ人物としてまとめられることを確認します。
        XCTAssertEqual(output.faces.first?.personID, person.id)
        // 不要な新人物が作られないことを確認します。
        XCTAssertTrue(output.people.isEmpty)
    // ここで直前の処理の範囲を閉じます。
    }

    // 人が除外した顔から次の時刻の人物を推測しないことを確認します。
    func testExcludedFaceCannotStartTemporalBridge() throws {
        // このテスト専用の保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite と画像を置くディレクトリを作ります。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 最後に生成ファイルを取り除きます。
        defer { try? FileManager.default.removeItem(at: directory) }
        // データベースの場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 初期データを書き込む接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 元フォルダを作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二秒から再開する動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("excluded.mp4").path, name: "excluded.mp4", duration: 4, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 2, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 人が除外した旧グループを作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: "excluded.jpg")
        // 除外済みの最初の顔を作ります。
        let old = FaceRecord(id: UUID(), videoID: video.id, second: 0, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.3), personID: person.id, thumbnailPath: "excluded.jpg", embedding: [1, 0], manualAssignment: true, excluded: true)
        // 登録元を保存します。
        try database.saveSource(source)
        // 動画を保存します。
        try database.saveVideo(video)
        // 旧グループを保存します。
        try database.savePerson(person)
        // 除外した顔を保存します。
        try database.saveFace(old)
        // 二秒後の顔を同じ位置に作ります。
        let detected = AnalyzedFace(second: 2, boundingBox: old.boundingBox!, thumbnailData: Data([2]), embedding: [0.69, 0.724])
        // 再開後の結果を作ります。
        let result = VideoAnalysisResult(duration: 4, posterData: nil, faces: [detected], completedSecond: 4, wasCancelled: false)
        // 除外済みの旧顔も読み取り専用の履歴に含めます。
        let input = FaceBatchInput(result: result, video: video, previousFaces: [old], representativeFaces: [old], databaseURL: databaseURL, thumbnailDirectory: directory)
        // 人物の自動割当を試します。
        let output = try FaceBatchStore.accept(input)
        // 除外した顔の人物を自動で引き継がないことを確認します。
        XCTAssertNotEqual(output.faces.first?.personID, person.id)
        // 除外された旧顔の印は保存されたままです。
        XCTAssertEqual(try database.load().faces.first(where: { $0.id == old.id })?.excluded, true)
    // ここで直前の処理の範囲を閉じます。
    }

    // 新しい区間だけの保存と失敗時のロールバックを検証します。
    func testFaceBatchesPreserveEarlierManualCorrectionsAndCheckpoint() throws {
        // 他のテストと共有しない一時保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite を作るためディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト終了時に一時データを取り除きます。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト専用の SQLite ファイルを開きます。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // 動画が所属する登録フォルダを作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二秒間隔で一部が解析済みの動画を作ります。
        var video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("sample.mp4").path, name: "sample.mp4", duration: 240, fileSize: 100, modificationTime: Date(), state: .pending, lastAnalyzedSecond: 120, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 人が統合先として選んだ人物を作ります。
        let person = PersonRecord(id: UUID(), name: "", thumbnailPath: "manual.jpg")
        // 最初の区間で人が割り当て直した顔を作ります。
        let corrected = FaceRecord(id: UUID(), videoID: video.id, second: 20, personID: person.id, thumbnailPath: "manual.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 登録フォルダを保存します。
        try database.saveSource(source)
        // 元動画を保存します。
        try database.saveVideo(video)
        // 手動割当先の人物を保存します。
        try database.savePerson(person)
        // 最初の区間の修正済み顔を保存します。
        try database.saveFace(corrected)
        // 次の区間で検出した顔を作ります。
        let nextFace = FaceRecord(id: UUID(), videoID: video.id, second: 122, personID: person.id, thumbnailPath: "next.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
        // 解析済み位置を二百秒まで進めます。
        video.lastAnalyzedSecond = 200
        // 次の区間だけ保存します。
        try database.applyFaceBatch(video: video, newPeople: [], faces: [nextFace], removing: [])
        // 新旧の両方の区間を読み直します。
        let saved = try database.load()
        // 前の区間の手動修正が残ることを確認します。
        XCTAssertEqual(saved.faces.first(where: { $0.id == corrected.id })?.manualAssignment, true)
        // 新しい区間の顔が追加されたことを確認します。
        XCTAssertTrue(saved.faces.contains { $0.id == nextFace.id })
        // 再開位置も顔と同時に保存されたことを確認します。
        XCTAssertEqual(saved.videos.first?.lastAnalyzedSecond, 200)
        // 存在しない人物への顔割当を作り、保存失敗を起こします。
        let invalid = FaceRecord(id: UUID(), videoID: video.id, second: 202, personID: UUID(), thumbnailPath: "invalid.jpg", embedding: [0, 1], manualAssignment: false, excluded: false)
        // 失敗する区間では完了位置をさらに進めます。
        video.lastAnalyzedSecond = 240
        // 外部キー違反で区間全体が取り消されることを確認します。
        XCTAssertThrowsError(try database.applyFaceBatch(video: video, newPeople: [], faces: [invalid], removing: [nextFace.id]))
        // 失敗後の状態を読み直します。
        let afterFailure = try database.load()
        // 失敗した区間が前の区間の顔を消していないことを確認します。
        XCTAssertTrue(afterFailure.faces.contains { $0.id == nextFace.id })
        // 失敗した区間の顔が保存されていないことを確認します。
        XCTAssertFalse(afterFailure.faces.contains { $0.id == invalid.id })
        // 再開位置も最後に成功した区間に戻ることを確認します。
        XCTAssertEqual(afterFailure.videos.first?.lastAnalyzedSecond, 200)
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
