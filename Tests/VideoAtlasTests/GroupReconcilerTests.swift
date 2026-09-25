// オフラインの人物グループ統合を検証します。
import XCTest
// アプリ内の照合器と保存型を利用します。
@testable import VideoAtlas

// 複数動画の顔だけで安全に統合できるか調べます。
final class GroupReconcilerTests: XCTestCase {
    // 旧ライブラリへ一度だけ自動統合する版情報を確認します。
    func testReconciliationVersionRunsOnceUntilIndexIsCleared() throws {
        // 他のデータと共有しない一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite ファイル用のディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に一時ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト専用の SQLite 接続を開きます。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // 新しいライブラリは未統合と判定します。
        XCTAssertFalse(try database.hasCurrentGroupReconciliation())
        // 現行版の照合完了を記録します。
        try database.markCurrentGroupReconciliation()
        // 再起動相当の再読込でも完了印が残ります。
        XCTAssertTrue(try database.hasCurrentGroupReconciliation())
        // インデックス消去で版情報も削除します。
        try database.clearAll()
        // 空にしたライブラリは再び未統合です。
        XCTAssertFalse(try database.hasCurrentGroupReconciliation())
    // ここで直前の処理の範囲を閉じます。
    }

    // 多数の一致で統合し、一組だけの偶然の一致と手動修正を保護します。
    func testCrossVideoConsensusMergesOnlySupportedAutomaticGroups() throws {
        // このテストだけの一時保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite を作れるディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に生成ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // データベースの場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // テスト用の SQLite 接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 三本の動画が所属する登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 各人物の顔を置く三本の動画を作ります。
        let videos = (0..<3).map { index in VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("\(index).mp4").path, name: "\(index).mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil) }
        // 強く一致する二組と弱い偶然の一組、手動修正済みの一組を作ります。
        let people = (0..<4).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 外部キーの親となる登録元を保存します。
        try database.saveSource(source)
        // 三本の動画を保存します。
        for video in videos { try database.saveVideo(video) }
        // 四人のグループを保存します。
        for person in people { try database.savePerson(person) }
        // 顔の一覧をまとめて作ります。
        var faces: [FaceRecord] = []
        // 複数の離れた時刻で顔を比較できるようにします。
        for index in 0..<20 {
            // 一つ目の動画にある人物の顔を作ります。
            let first = FaceRecord(id: UUID(), videoID: videos[0].id, second: Double(index * 2), personID: people[0].id, thumbnailPath: "a.jpg", embedding: [1, 0, 0], manualAssignment: false, excluded: false)
            // 二つ目の動画にある同じ特徴の顔を作ります。
            let matching = FaceRecord(id: UUID(), videoID: videos[1].id, second: Double(index * 2), personID: people[1].id, thumbnailPath: "b.jpg", embedding: [0.99, 0.1, 0], manualAssignment: false, excluded: false)
            // 三つ目のグループは一枚だけ偶然に近くします。
            let accidental = FaceRecord(id: UUID(), videoID: videos[2].id, second: Double(index * 2), personID: people[2].id, thumbnailPath: "c.jpg", embedding: index == 0 ? [1, 0, 0] : [0, 1, 0], manualAssignment: false, excluded: false)
            // 四つ目のグループは近くても一枚を人が修正済みとします。
            let manual = FaceRecord(id: UUID(), videoID: videos[2].id, second: Double(index * 2 + 1), personID: people[3].id, thumbnailPath: "d.jpg", embedding: [1, 0, 0], manualAssignment: index == 0, excluded: false)
            // 四組の顔を保存対象へ追加します。
            faces.append(contentsOf: [first, matching, accidental, manual])
        // ここで直前の処理の範囲を閉じます。
        }
        // 旧状態をデータベースへ保存します。
        for face in faces { try database.saveFace(face) }
        // 四組の情報を背景照合器へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: videos, people: people, databaseURL: databaseURL)
        // 顔の複数時刻による根拠を比較し、必要な変更だけ保存します。
        let output = try GroupReconciler.reconcile(input)
        // 強く一致する二組だけで統合元が一件になることを確かめます。
        XCTAssertEqual(output.personMapping.count, 1)
        // 保存後の顔と人物を読み直します。
        let saved = try database.load()
        // 一つ目と二つ目の動画で同じ人物 ID が使われます。
        let firstIDs = Set(saved.faces.filter { $0.videoID == videos[0].id }.compactMap(\.personID))
        // 二つ目の動画の人物 ID を集めます。
        let secondIDs = Set(saved.faces.filter { $0.videoID == videos[1].id }.compactMap(\.personID))
        // 二つの動画で一人の顔写真へまとまったことを確認します。
        XCTAssertEqual(firstIDs, secondIDs)
        // 偶然の一枚しか一致しない人物は統合されません。
        XCTAssertTrue(saved.faces.contains { $0.personID == people[2].id })
        // 手動修正済みの人物は自動で変更されません。
        XCTAssertTrue(saved.faces.contains { $0.personID == people[3].id && $0.manualAssignment })
        // 元顔の手動修正フラグを持つ件数は維持されます。
        XCTAssertEqual(saved.faces.filter(\.manualAssignment).count, 1)
    // ここで直前の処理の範囲を閉じます。
    }

    // 三十二枚の均等サンプルが強い一組を逃しても全顔の根拠で統合します。
    func testCoarseCandidateUsesFullFaceConsensus() throws {
        // この検証だけの一時保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite を作れるディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に生成ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // SQLite の場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // テスト用の接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 二本の動画の登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 異なる動画に分かれた二組を作ります。
        let videos = (0..<2).map { index in VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("sampled-\(index).mp4").path, name: "sampled-\(index).mp4", duration: 500, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 500, sampleInterval: 2, errorMessage: nil, posterPath: nil) }
        // 同じ人物が別々のグループになった状態を作ります。
        let people = (0..<2).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 外部キーの親となる登録元を保存します。
        try database.saveSource(source)
        // 二本の動画を保存します。
        for video in videos { try database.saveVideo(video) }
        // 二組の人物を保存します。
        for person in people { try database.savePerson(person) }
        // 以前の均等選抜で二本目から選ばれる三十二の添字を計算します。
        let oldSampleIndices = Set((0..<32).map { Int((Double($0) * 199 / 31).rounded()) })
        // 全顔を照合器へ渡す配列を作ります。
        var faces: [FaceRecord] = []
        // 一本目には百時刻分の同じ特徴を置きます。
        for index in 0..<100 {
            // 一本目の顔を作ります。
            let face = FaceRecord(id: UUID(), videoID: videos[0].id, second: Double(index * 2), personID: people[0].id, thumbnailPath: "a.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
            // 保存対象へ追加します。
            faces.append(face)
        // ここで直前の処理の範囲を閉じます。
        }
        // 二本目には二百時刻分の顔を置きます。
        for index in 0..<200 {
            // 均等抽出された顔だけ少し違う角度にします。
            let embedding: [Float] = oldSampleIndices.contains(index) ? [0.78, 0.626] : [0.84, 0.543]
            // 二本目の顔を作ります。
            let face = FaceRecord(id: UUID(), videoID: videos[1].id, second: Double(index * 2), personID: people[1].id, thumbnailPath: "b.jpg", embedding: embedding, manualAssignment: false, excluded: false)
            // 保存対象へ追加します。
            faces.append(face)
        // ここで直前の処理の範囲を閉じます。
        }
        // 元の三百件の顔を保存します。
        for face in faces { try database.saveFace(face) }
        // 全顔と動画を背景照合へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: videos, people: people, databaseURL: databaseURL)
        // 緩い三十二枚の候補選抜後に全顔の厳しい根拠を計算します。
        let output = try GroupReconciler.reconcile(input)
        // 均等三十二枚では最短 0.22 でも全顔なら 0.16 のため統合されます。
        XCTAssertEqual(output.personMapping.count, 1)
        // 最終保存状態では一人の人物 ID だけが残ります。
        XCTAssertEqual(Set(try database.load().faces.compactMap(\.personID)).count, 1)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同一動画の断片より先に別動画との強い一致を確定します。
    func testCrossVideoMatchTakesPriorityOverSameVideoFragments() throws {
        // 検証専用の一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite を置くディレクトリを作ります。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に生成ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // データベースの場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 登録情報を保存する接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 二本の動画の登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 一方の動画には三つの断片、もう一方には一致する一組を置きます。
        let videos = (0..<2).map { index in VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("priority-\(index).mp4").path, name: "priority-\(index).mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil) }
        // 四つに分かれた人物グループを作ります。
        let people = (0..<4).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 顔より先に登録元を保存します。
        try database.saveSource(source)
        // 二本の動画を保存します。
        for video in videos { try database.saveVideo(video) }
        // 四つの人物グループを保存します。
        for person in people { try database.savePerson(person) }
        // 最初の三組は同じ動画内で近く、四つ目は第一組だけに近くします。
        let embeddings: [[Float]] = [[1, 0], [0.87758, -0.47943], [0.90045, -0.43497], [0.82534, 0.56464]]
        // 各グループの複数時刻の顔を保持します。
        var faces: [FaceRecord] = []
        // 四組を一つずつ作ります。
        for groupIndex in 0..<4 {
            // 最後の組だけ別動画に置きます。
            let videoID = videos[groupIndex == 3 ? 1 : 0].id
            // 同じ動画内の三組が同時刻にならないようずらします。
            let offset = Double(groupIndex == 3 ? 0 : groupIndex) * 0.5
            // 各組に二十時刻の顔を用意します。
            for index in 0..<20 {
                // 今回の時刻と特徴量を持つ顔を作ります。
                let face = FaceRecord(id: UUID(), videoID: videoID, second: Double(index * 2) + offset, personID: people[groupIndex].id, thumbnailPath: "priority.jpg", embedding: embeddings[groupIndex], manualAssignment: false, excluded: false)
                // 顔一覧へ加えます。
                faces.append(face)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 元の分類済み顔を保存します。
        for face in faces { try database.saveFace(face) }
        // 四組を自動照合へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: videos, people: people, databaseURL: databaseURL)
        // 跨動画の辺を先に確定してから同じ動画の断片を検討します。
        let output = try GroupReconciler.reconcile(input)
        // 保存された顔を人物ごとに読み直します。
        let saved = try database.load().faces
        // 各元グループの代表顔を最終状態から探します。
        let finalIDs = people.map { person in saved.first(where: { $0.id == faces.first(where: { $0.personID == person.id })!.id })!.personID }
        // 第一組と別動画の第四組が同じ人物になります。
        XCTAssertEqual(finalIDs[0], finalIDs[3])
        // 第二組は別動画への根拠がないため第一組へ連鎖しません。
        XCTAssertNotEqual(finalIDs[0], finalIDs[1])
        // 第三組も別動画への根拠がないため第一組へ連鎖しません。
        XCTAssertNotEqual(finalIDs[0], finalIDs[2])
        // 少なくとも一組の跨動画統合が記録されます。
        XCTAssertFalse(output.personMapping.isEmpty)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同じ動画内の別時刻なら強い根拠で統合できます。
    func testSameVideoRequiresStrongerConsensus() throws {
        // この検証のためだけのディレクトリを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite 用の場所を用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に一時ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // データベースの場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // テスト用の接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 元フォルダを作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二つのグループが異なる時刻に現れる動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("one.mp4").path, name: "one.mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 別々に分類された二人を作ります。
        let people = (0..<2).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 登録元を保存します。
        try database.saveSource(source)
        // 動画を保存します。
        try database.saveVideo(video)
        // 二人を保存します。
        for person in people { try database.savePerson(person) }
        // 十五時刻ずつの顔を保存する配列を作ります。
        var faces: [FaceRecord] = []
        // 異なるフレームで同じ特徴が十分続く二組を作ります。
        for index in 0..<15 {
            // 偶数秒に映る一つ目の顔を作ります。
            let first = FaceRecord(id: UUID(), videoID: video.id, second: Double(index * 2), personID: people[0].id, thumbnailPath: "one.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
            // その一秒後に映る二つ目の顔を作ります。
            let second = FaceRecord(id: UUID(), videoID: video.id, second: Double(index * 2 + 1), personID: people[1].id, thumbnailPath: "two.jpg", embedding: [0.99, 0.1], manualAssignment: false, excluded: false)
            // 二件とも保存対象へ追加します。
            faces.append(contentsOf: [first, second])
        // ここで直前の処理の範囲を閉じます。
        }
        // 元の顔を保存します。
        for face in faces { try database.saveFace(face) }
        // 同じ動画の二組を照合器へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: [video], people: people, databaseURL: databaseURL)
        // 厳しい同一動画の基準で比較します。
        let output = try GroupReconciler.reconcile(input)
        // 十件以上の双方向支持によって統合されます。
        XCTAssertEqual(output.personMapping.count, 1)
        // 保存後は一つの人物 ID だけが残ります。
        XCTAssertEqual(Set(try database.load().faces.compactMap(\.personID)).count, 1)
    // ここで直前の処理の範囲を閉じます。
    }

    // 二つの強い辺だけでは三人を連鎖して統合しません。
    func testBridgeWithoutEvidenceBetweenEndpointsDoesNotMergeAllThree() throws {
        // 他のテストと分けた一時保存先を作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite を置くディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に一時ファイルを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト用データベースの場所を決めます。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 元データを書き込む接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 三本の動画の登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 各グループが別の動画に映るようにします。
        let videos = (0..<3).map { index in VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("chain-\(index).mp4").path, name: "chain-\(index).mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil) }
        // 三つの元人物グループを作ります。
        let people = (0..<3).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 登録元を保存します。
        try database.saveSource(source)
        // 三本の動画を保存します。
        for video in videos { try database.saveVideo(video) }
        // 三人を保存します。
        for person in people { try database.savePerson(person) }
        // A と B、B と C は近く、A と C は離れた特徴量です。
        let embeddings: [[Float]] = [[1, 0], [0.9, 0.43589], [0.62, 0.7846]]
        // 各グループの二十時刻分の顔を作ります。
        var faces: [FaceRecord] = []
        // 三つの元グループを順番に作ります。
        for groupIndex in 0..<3 {
            // 一つのグループの二十時刻を作ります。
            for index in 0..<20 {
                // 今回の人物の顔を作ります。
                let face = FaceRecord(id: UUID(), videoID: videos[groupIndex].id, second: Double(index * 2), personID: people[groupIndex].id, thumbnailPath: "chain.jpg", embedding: embeddings[groupIndex], manualAssignment: false, excluded: false)
                // 後で照合できるよう配列へ追加します。
                faces.append(face)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 元の顔を SQLite へ保存します。
        for face in faces { try database.saveFace(face) }
        // 三組の情報を照合器へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: videos, people: people, databaseURL: databaseURL)
        // 全組の根拠を要求して統合します。
        let output = try GroupReconciler.reconcile(input)
        // 根拠のない端点同士までつながらず統合元は一件だけです。
        XCTAssertEqual(output.personMapping.count, 1)
        // 最終的な人物 ID は二つ残ります。
        XCTAssertEqual(Set(try database.load().faces.compactMap(\.personID)).count, 2)
    // ここで直前の処理の範囲を閉じます。
    }

    // 同時に映る二組の顔はどれだけ似ていても統合しません。
    func testSimultaneousGroupsRemainSeparate() throws {
        // 他の検証と分けた一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // データベースを作るディレクトリを用意します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に生成ファイルを削除します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // SQLite のファイル場所を作ります。
        let databaseURL = directory.appendingPathComponent("library.sqlite")
        // 元データ用の接続を開きます。
        let database = try LibraryDatabase(databaseURL: databaseURL)
        // 動画の登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 二人が同時に映る動画を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("two.mp4").path, name: "two.mp4", duration: 100, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 100, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 似た特徴量の二人を作ります。
        let people = (0..<2).map { _ in PersonRecord(id: UUID(), name: "", thumbnailPath: nil) }
        // 登録元を保存します。
        try database.saveSource(source)
        // 動画を保存します。
        try database.saveVideo(video)
        // 二人を保存します。
        for person in people { try database.savePerson(person) }
        // 同時刻の顔を保存する配列を用意します。
        var faces: [FaceRecord] = []
        // 二十フレームで二人が同時に映る状態を作ります。
        for index in 0..<20 {
            // 一つ目の顔を作ります。
            let first = FaceRecord(id: UUID(), videoID: video.id, second: Double(index * 2), personID: people[0].id, thumbnailPath: "one.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
            // 同じフレームの別の顔を作ります。
            let second = FaceRecord(id: UUID(), videoID: video.id, second: Double(index * 2), personID: people[1].id, thumbnailPath: "two.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
            // 二人を今回の顔一覧へ加えます。
            faces.append(contentsOf: [first, second])
        // ここで直前の処理の範囲を閉じます。
        }
        // 元の顔をすべて保存します。
        for face in faces { try database.saveFace(face) }
        // 同時出現の情報を照合器へ渡します。
        let input = GroupReconciliationInput(faces: faces, videos: [video], people: people, databaseURL: databaseURL)
        // 自動統合の候補を調べます。
        let output = try GroupReconciler.reconcile(input)
        // 同時出現した二組を統合しないことを確認します。
        XCTAssertTrue(output.personMapping.isEmpty)
        // 人物グループが二つ残ることを確認します。
        XCTAssertEqual(try database.load().people.count, 2)
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
