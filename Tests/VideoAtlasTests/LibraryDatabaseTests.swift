// XCTest の検証機能を読み込みます。
import XCTest
// アプリ内のデータ型とデータベースをテストから使います。
@testable import VideoAtlas
// SQLite に保存した情報が再読込後にも残ることを確認します。
final class LibraryDatabaseTests: XCTestCase {
    // フォルダ・動画・人物・顔の保存と関連削除を検証します。
    func testRoundTripAndCascadeRemoval() throws {
        // 他のテストと共有しない一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // 一時フォルダをディスク上に作成します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト後に一時フォルダを削除します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // 一時フォルダに SQLite データベースを作ります。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // テスト用のフォルダ情報を作ります。
        let source = SourceFolder(id: UUID(), path: "/tmp/videos", bookmark: Data([1, 2, 3]))
        // テスト用の動画情報を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: "/tmp/videos/a.mp4", name: "a.mp4", duration: 6, fileSize: 100, modificationTime: Date(timeIntervalSince1970: 100), state: .ready, lastAnalyzedSecond: 6, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // テスト用の人物情報を作ります。
        let person = PersonRecord(id: UUID(), name: "人物 A", thumbnailPath: nil)
        // テスト用の顔検出情報を作ります。
        let face = FaceRecord(id: UUID(), videoID: video.id, second: 2, boundingBox: CGRect(x: 0.2, y: 0.3, width: 0.2, height: 0.3), personID: person.id, thumbnailPath: "face.jpg", embedding: [0.1, 0.2], manualAssignment: true, excluded: false)
        // 外部キーの親となるフォルダを保存します。
        try database.saveSource(source)
        // フォルダに属する動画を保存します。
        try database.saveVideo(video)
        // 顔が参照する人物を保存します。
        try database.savePerson(person)
        // 顔と特徴量を保存します。
        try database.saveFace(face)
        // SQLite に保存した全情報を読み戻します。
        let loaded = try database.load()
        // フォルダの ID が保存されていることを確認します。
        XCTAssertEqual(loaded.sources.map(\.id), [source.id])
        // 動画の ID が保存されていることを確認します。
        XCTAssertEqual(loaded.videos.map(\.id), [video.id])
        // 人物の ID が保存されていることを確認します。
        XCTAssertEqual(loaded.people.map(\.id), [person.id])
        // 顔の時刻と手動設定が残ることを確認します。
        XCTAssertEqual(loaded.faces.first?.second, 2)
        // 顔の手動割当が残ることを確認します。
        XCTAssertEqual(loaded.faces.first?.manualAssignment, true)
        // 再解析で顔を対応付ける位置情報も残ることを確認します。
        XCTAssertEqual(loaded.faces.first?.boundingBox?.origin.x, 0.2)
        // フォルダを削除して関連動画と顔の連鎖削除を試します。
        try database.removeSource(id: source.id)
        // 削除後の状態を読み直します。
        let remaining = try database.load()
        // 元の動画が消えていることを確認します。
        XCTAssertTrue(remaining.videos.isEmpty)
        // 動画に属した顔も消えていることを確認します。
        XCTAssertTrue(remaining.faces.isEmpty)
    // ここで直前の処理の範囲を閉じます。
    }

    // 動画の内容変更では旧顔を消し、他の動画が使う人物を残します。
    func testChangedVideoRevisionInvalidatesOnlyItsOldFaces() throws {
        // 他のテストと共有しない一時ディレクトリを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // SQLite ファイルを置けるようにします。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // 検証後に一時データを削除します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト専用のデータベースを開きます。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // 二つの動画に共通する登録元を作ります。
        let source = SourceFolder(id: UUID(), path: directory.path, bookmark: Data([1]))
        // 内容を更新する動画を作ります。
        let changed = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("changed.mp4").path, name: "changed.mp4", duration: 10, fileSize: 100, modificationTime: Date(timeIntervalSince1970: 100), state: .ready, lastAnalyzedSecond: 10, sampleInterval: 2, errorMessage: nil, posterPath: "old-poster.jpg")
        // 変更しない別の動画を作ります。
        let other = VideoRecord(id: UUID(), sourceID: source.id, path: directory.appendingPathComponent("other.mp4").path, name: "other.mp4", duration: 10, fileSize: 200, modificationTime: Date(timeIntervalSince1970: 100), state: .ready, lastAnalyzedSecond: 10, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 更新動画だけに現れる人物を作ります。
        let oldOnly = PersonRecord(id: UUID(), name: "", thumbnailPath: nil)
        // 二つの動画に現れる人物を作ります。
        let shared = PersonRecord(id: UUID(), name: "", thumbnailPath: nil)
        // 更新動画だけの顔に手動修正を記録します。
        let oldFace = FaceRecord(id: UUID(), videoID: changed.id, second: 2, personID: oldOnly.id, thumbnailPath: "old-face.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 更新動画で共通人物に属する顔を作ります。
        let sharedOldFace = FaceRecord(id: UUID(), videoID: changed.id, second: 4, personID: shared.id, thumbnailPath: "shared-old.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 変更しない動画の顔を作ります。
        let otherFace = FaceRecord(id: UUID(), videoID: other.id, second: 2, personID: shared.id, thumbnailPath: "other-face.jpg", embedding: [1, 0], manualAssignment: true, excluded: false)
        // 先に登録元を保存します。
        try database.saveSource(source)
        // 更新前の動画情報を保存します。
        try database.saveVideo(changed)
        // 別の動画情報を保存します。
        try database.saveVideo(other)
        // 更新動画だけの人物を保存します。
        try database.savePerson(oldOnly)
        // 共通人物を保存します。
        try database.savePerson(shared)
        // 更新動画だけの顔を保存します。
        try database.saveFace(oldFace)
        // 更新動画にある共通人物の顔を保存します。
        try database.saveFace(sharedOldFace)
        // 別動画の顔を保存します。
        try database.saveFace(otherFace)
        // 新しいファイル属性を保持するレコードを作ります。
        var revision = changed
        // ファイルサイズの変更を記録します。
        revision.fileSize = 101
        // 最後に処理した時刻を消します。
        revision.lastAnalyzedSecond = 0
        // 古い再生時間を消します。
        revision.duration = 0
        // 古いポスターの参照を消します。
        revision.posterPath = nil
        // 再解析待ちにします。
        revision.state = .pending
        // 動画更新と旧顔削除を一括保存します。
        let removedPeople = try database.invalidateVideoRevision(revision)
        // 更新動画だけに属した人物のみ削除されたことを確かめます。
        XCTAssertEqual(removedPeople, Set([oldOnly.id]))
        // DB を読み直して永続状態を確認します。
        let loaded = try database.load()
        // 別動画の顔だけが残ることを確かめます。
        XCTAssertEqual(loaded.faces.map(\.id), [otherFace.id])
        // 共通人物が残ることを確かめます。
        XCTAssertEqual(loaded.people.map(\.id), [shared.id])
        // 新しいサイズが保存されたことを確かめます。
        XCTAssertEqual(loaded.videos.first(where: { $0.id == changed.id })?.fileSize, 101)
        // 旧解析位置が残らないことを確かめます。
        XCTAssertEqual(loaded.videos.first(where: { $0.id == changed.id })?.lastAnalyzedSecond, 0)
        // 旧ポスターの参照が残らないことを確かめます。
        XCTAssertNil(loaded.videos.first(where: { $0.id == changed.id })?.posterPath)
    // ここで直前の処理の範囲を閉じます。
    }

    // 人が直した人物グループを再起動後にも読み戻せることを確かめます。
    @MainActor func testManualChangesSurviveRestart() throws {
        // 他のライブラリに影響しない一時フォルダを作ります。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // データベース用のフォルダを作成します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テストの終了後に一時データだけを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // 事前データを書くため SQLite を開きます。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // 登録済みフォルダの代わりとなる情報を作ります。
        let source = SourceFolder(id: UUID(), path: "/tmp/videoatlas-test", bookmark: Data([1]))
        // 二つの顔を持つ動画情報を作ります。
        let video = VideoRecord(id: UUID(), sourceID: source.id, path: "/tmp/videoatlas-test/a.mp4", name: "a.mp4", duration: 8, fileSize: 100, modificationTime: Date(), state: .ready, lastAnalyzedSecond: 8, sampleInterval: 2, errorMessage: nil, posterPath: nil)
        // 統合元の人物を作ります。
        let firstPerson = PersonRecord(id: UUID(), name: "人物 1", thumbnailPath: nil)
        // 統合先の人物を作ります。
        let secondPerson = PersonRecord(id: UUID(), name: "人物 2", thumbnailPath: nil)
        // 統合元に属する一つ目の顔を作ります。
        let firstFace = FaceRecord(id: UUID(), videoID: video.id, second: 2, personID: firstPerson.id, thumbnailPath: "first.jpg", embedding: [1, 0], manualAssignment: false, excluded: false)
        // 統合先に属する二つ目の顔を作ります。
        let secondFace = FaceRecord(id: UUID(), videoID: video.id, second: 4, personID: secondPerson.id, thumbnailPath: "second.jpg", embedding: [0, 1], manualAssignment: false, excluded: false)
        // フォルダを先に保存します。
        try database.saveSource(source)
        // フォルダ内の動画を保存します。
        try database.saveVideo(video)
        // 統合元の人物を保存します。
        try database.savePerson(firstPerson)
        // 統合先の人物を保存します。
        try database.savePerson(secondPerson)
        // 一つ目の顔を保存します。
        try database.saveFace(firstFace)
        // 二つ目の顔を保存します。
        try database.saveFace(secondFace)
        // 保存済み状態からアプリモデルを起動します。
        let model = AppModel(storageDirectory: directory)
        // 統合先の名前を人が変更します。
        model.renamePerson(id: secondPerson.id, name: "Pat")
        // 統合元の顔を統合先へ移します。
        model.mergePeople(sourceID: firstPerson.id, targetID: secondPerson.id)
        // 一つ目の顔を新しい人物として分割します。
        model.splitFace(faceID: firstFace.id)
        // 二つ目の顔を誤登録として除外します。
        model.excludeFace(faceID: secondFace.id)
        // 再起動を模して同じ SQLite を別のモデルで開きます。
        let restarted = AppModel(storageDirectory: directory)
        // 統合元が消えたことを確認します。
        XCTAssertFalse(restarted.people.contains { $0.id == firstPerson.id })
        // 変更した名前が残ることを確認します。
        XCTAssertEqual(restarted.people.first(where: { $0.id == secondPerson.id })?.name, "Pat")
        // 分割した顔が新しい人物に割り当てられていることを確認します。
        XCTAssertNotEqual(restarted.faces.first(where: { $0.id == firstFace.id })?.personID, secondPerson.id)
        // 分割は手動設定として保存されていることを確認します。
        XCTAssertEqual(restarted.faces.first(where: { $0.id == firstFace.id })?.manualAssignment, true)
        // 除外した顔の設定が残ることを確認します。
        XCTAssertEqual(restarted.faces.first(where: { $0.id == secondFace.id })?.excluded, true)
    // ここで直前の処理の範囲を閉じます。
    }

    // 読み取り権限を解決できないフォルダを利用者へ通知します。
    @MainActor func testInvalidFolderPermissionIsReported() throws {
        // 実際のライブラリを触らない一時ディレクトリを選びます。
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // 一時ディレクトリを作成します。
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // テスト終了時に一時データを消します。
        defer { try? FileManager.default.removeItem(at: directory) }
        // テスト用データベースを作ります。
        let database = try LibraryDatabase(databaseURL: directory.appendingPathComponent("library.sqlite"))
        // 壊れたブックマークを持つフォルダを作ります。
        let source = SourceFolder(id: UUID(), path: "/tmp/videoatlas-no-permission", bookmark: Data([0, 1, 2]))
        // 権限切れに相当する登録情報を保存します。
        try database.saveSource(source)
        // 保存済みフォルダをモデルへ読み込みます。
        let model = AppModel(storageDirectory: directory)
        // 登録フォルダへのアクセスを試します。
        model.refreshLibrary()
        // 解析済みと誤表示せず、確認できない理由を伝えることを確かめます。
        XCTAssertTrue(model.statusText.contains("確認できません"))
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
