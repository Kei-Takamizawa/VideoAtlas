// 人物一覧と動画を表示するために SwiftUI を読み込みます。
import SwiftUI
// 顔画像のファイルを開いて表示するために AppKit を読み込みます。
import AppKit

// 顔を見つけた時刻から再生画面を開くための情報をまとめます。
private struct PersonMoment: Identifiable {
    // 顔レコードの ID をシート表示の識別子として使います。
    let id: UUID
    // 再生する動画の情報を保持します。
    let video: VideoRecord
    // 顔を検出した秒数を保持します。
    let second: Double
// この構造体の定義を閉じます。
}

// 顔写真を選んで同じ人物が映る動画を探す画面です。
struct PeopleView: View {
    // アプリ全体で共有する解析済みデータを受け取ります。
    var model: AppModel
    // 選択中の人物グループ ID を保持します。
    @State private var selectedPersonID: UUID?
    // 未分類の顔を選んでいるかを保持します。
    @State private var showsUnclassified = false
    // 検出時刻を押した後に開く再生情報を保持します。
    @State private var selectedMoment: PersonMoment?
    // 検出時刻が多い動画でも最初は 60 件だけ表示するため件数を保持します。
    @State private var visibleFaceCounts: [UUID: Int] = [:]
    // 顔写真の一覧と動画結果を左右に配置します。
    var body: some View {
        // 見出しと解析操作を上部に、顔一覧と結果を下部に配置します。
        VStack(alignment: .leading, spacing: 16) {
            // 画面名と共有解析コントロールを横一列に並べます。
            HStack {
                // 人物画面の見出しを表示します。
                Text("人物").font(.largeTitle.bold())
                // 解析操作を右端へ寄せます。
                Spacer()
                // 解析状態と一時停止または再開操作を表示します。
                scanControls
            // 見出し行の定義を閉じます。
            }
            // 左側の顔選択欄と右側の結果欄を横に並べます。
            HStack(alignment: .top, spacing: 24) {
                // 人物グループを顔写真だけで選べる一覧を表示します。
                faceGallery.frame(width: 300)
                // 選んだ人物の動画一覧と検出時刻を表示します。
                results.frame(maxWidth: .infinity, maxHeight: .infinity)
            // 横並びレイアウトの定義を閉じます。
            }
        // 見出しと内容の縦並びを閉じます。
        }
        // 画面の周囲に操作しやすい余白を付けます。
        .padding(24)
        // 検出時刻を選んだときに動画プレイヤーを開きます。
        .sheet(item: $selectedMoment) { moment in
            // 選択時刻から動画を再生できる画面を表示します。
            VideoPlayerView(model: model, video: moment.video, initialSecond: moment.second)
        // シート内容の定義を閉じます。
        }
    // 画面本体の定義を閉じます。
    }
    // 人物画面にも解析状態と停止・再開操作を表示します。
    @ViewBuilder private var scanControls: some View {
        // 解析中は状態文字、進捗、停止ボタンを表示します。
        if model.isScanning {
            // 進捗と一時停止操作を狭い幅にまとめます。
            HStack(spacing: 10) {
                // 現在処理中の動画名を省略して表示します。
                Text(model.statusText).font(.caption).foregroundStyle(.secondary).lineLimit(1).frame(maxWidth: 240, alignment: .trailing)
                // ライブラリ全体の解析進捗を表示します。
                ProgressView(value: model.progress).frame(width: 90)
                // 現在の動画を安全に保存して解析を止めます。
                Button("一時停止") { model.pauseScanning() }
            // 解析中コントロールの横並びを閉じます。
            }
        // 解析待ち動画がある場合は保存済みの位置から再開する操作を表示します。
        } else if model.videos.contains(where: { $0.state == .pending }) {
            // 停止状態の説明と解析再開操作を横一列に並べます。
            HStack(spacing: 10) {
                // 一時停止など現在の状態を表示します。
                Text(model.statusText).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                // 待機中の動画解析だけを再開します。
                Button("解析を再開") { model.resumeScanning() }
            // 再開コントロールの横並びを閉じます。
            }
        // 解析状態の説明があれば完了やエラーの情報を表示します。
        } else if !model.statusText.isEmpty {
            // 完了またはエラーの状態を控えめに表示します。
            Text(model.statusText).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        // 状態表示の条件を閉じます。
        }
    // 解析コントロールの定義を閉じます。
    }
    // 顔写真をタイル状に並べる選択欄を作ります。
    private var faceGallery: some View {
        // 見出し、未分類の入口、顔写真一覧を縦に並べます。
        VStack(alignment: .leading, spacing: 14) {
            // 未分類の顔をまとめて選べるボタンを作ります。
            Button {
                // 特定人物の選択を解除します。
                selectedPersonID = nil
                // 未分類一覧を表示する状態にします。
                showsUnclassified = true
            // 未分類ボタンの操作を閉じます。
            } label: {
                // 未分類の顔があることを文字とアイコンで示します。
                Label("未分類の顔", systemImage: "person.crop.circle.badge.questionmark")
            // ボタン表示の定義を閉じます。
            }
            // Liquid Glass のボタンで切り替え操作を表します。
            .buttonStyle(.glass)
            // すべての人物グループをスクロール可能な領域へ並べます。
            ScrollView {
                // 顔写真を二列のタイルで整列させます。
                LazyVGrid(columns: [GridItem(.flexible(), spacing: 12), GridItem(.flexible(), spacing: 12)], spacing: 12) {
                    // 登録済みの人物グループを顔写真で表します。
                    ForEach(model.people) { person in
                        // 写真を押すとその人物の動画へ絞り込みます。
                        Button {
                            // 選択人物をタイルの人物 ID に更新します。
                            selectedPersonID = person.id
                            // 未分類表示を閉じます。
                            showsUnclassified = false
                        // 人物写真タイルの操作を閉じます。
                        } label: {
                            // 写真と背景だけで人物を判別できるようにします。
                            personPortrait(person, size: 112)
                        // タイル内容の定義を閉じます。
                        }
                        // ボタンの標準余白をなくして写真を押しやすくします。
                        .buttonStyle(.plain)
                        // 選択写真に枠を付け、選択状態を伝えます。
                        .padding(4).overlay(RoundedRectangle(cornerRadius: 18).stroke(selectedPersonID == person.id && !showsUnclassified ? Color.accentColor : Color.clear, lineWidth: 3))
                        // VoiceOver には写真選択の機能だけを伝えます。
                        .accessibilityLabel("人物の顔写真")
                        // 写真を右クリックしたときの修正操作を表示します。
                        .contextMenu {
                            // 他の人物グループを統合先として並べます。
                            ForEach(model.people.filter { $0.id != person.id }) { target in
                                // 統合先の顔写真を選ぶと現在のグループをまとめます。
                                Button {
                                    // 元グループを写真で選んだグループへ統合します。
                                    model.mergePeople(sourceID: person.id, targetID: target.id)
                                    // 統合後も選んだ顔写真を表示対象にします。
                                    selectedPersonID = target.id
                                // 統合操作の範囲を閉じます。
                                } label: {
                                    // 統合先の顔を見て選べるメニュー項目にします。
                                    HStack {
                                        // 統合先の代表顔写真を選択肢に添えます。
                                        personPortrait(target, size: 28)
                                        // 顔写真を現在のグループへまとめる操作だと説明します。
                                        Text("この顔のグループにまとめる")
                                    // 顔写真と操作説明を横に並べる処理を閉じます。
                                    }
                                // メニュー項目の表示を閉じます。
                                }
                                // 解析中は統合操作を無効にして共有データの競合を防ぎます。
                                .disabled(model.isScanning)
                                // 顔写真の有無または解析停止が必要なことを説明します。
                                .help(model.isScanning ? "顔の修正は解析を一時停止してから行えます" : (target.thumbnailPath == nil ? "代表顔写真がありません" : "表示中の顔写真のグループへ統合します"))
                            // 統合先の一覧を閉じます。
                            }
                        // コンテキストメニューを閉じます。
                        }
                    // 人物ごとのタイル生成を閉じます。
                    }
                // 顔写真グリッドを閉じます。
                }
                // 写真が端に寄りすぎないよう余白を付けます。
                .padding(4)
            // スクロール領域を閉じます。
            }
            // 顔写真を含まない項目が多い場合も一覧を縦に伸ばします。
            .frame(maxHeight: .infinity)
        // ギャラリーの縦並びを閉じます。
        }
    // 顔写真一覧欄の定義を閉じます。
    }
    // 選択した人物に関連する動画と顔の検出時刻を表示します。
    @ViewBuilder private var results: some View {
        // 選択顔を動画 ID ごとに一度だけまとめ、各カードの再フィルターを防ぎます。
        let groupedFaces = Dictionary(grouping: selectedFaces, by: \.videoID)
        // 顔グループに含まれる動画だけをライブラリ順で並べます。
        let resultVideos = model.videos.filter { groupedFaces[$0.id] != nil }
        // 人物写真か未分類項目が選ばれている場合に結果を作ります。
        if selectedPersonID != nil || showsUnclassified {
            // 結果の見出しと動画一覧を縦に並べます。
            VStack(alignment: .leading, spacing: 16) {
                // 選択した顔写真と動画件数を見出しにします。
                HStack(spacing: 12) {
                    // 未分類を選んだときは説明アイコンを表示します。
                    if showsUnclassified {
                        // 未分類の顔を示すアイコンを大きく表示します。
                        Image(systemName: "person.crop.circle.badge.questionmark").font(.system(size: 38)).foregroundStyle(.secondary)
                    // 人物グループが選ばれているときは代表写真を表示します。
                    } else if let person = selectedPerson {
                        // 選択した人物を名前や番号なしで示します。
                        personPortrait(person, size: 56)
                    // 選択情報が一時的に見つからない場合は記号を表示します。
                    } else {
                        // 情報なしの状態を控えめなアイコンで示します。
                        Image(systemName: "person.crop.circle").font(.system(size: 38)).foregroundStyle(.secondary)
                    // 顔写真表示の条件を閉じます。
                    }
                    // 動画件数だけを説明する見出しを表示します。
                    Text("登場する動画 · \(resultVideos.count) 件").font(.title2.bold())
                // 見出しの横並びを閉じます。
                }
                // 対象動画がない場合に空状態を説明します。
                if resultVideos.isEmpty {
                    // 該当動画がまだないことを伝えます。
                    ContentUnavailableView("該当する動画がありません", systemImage: "film.stack")
                // 動画が見つかった場合はカード一覧を表示します。
                } else {
                    // 動画数が多くてもスクロールして探せるようにします。
                    ScrollView {
                        // 動画カードを読みやすい間隔で縦に並べます。
                        LazyVStack(alignment: .leading, spacing: 14) {
                            // 同じ人物が検出された動画を順番に表示します。
                            ForEach(resultVideos) { video in
                                // 動画名と顔検出時刻の一覧を表示します。
                                videoResult(video, faces: groupedFaces[video.id] ?? [])
                            // 動画一覧の繰り返しを閉じます。
                            }
                        // 動画カードの縦並びを閉じます。
                        }
                        // 最後のカードの下に余白を設けます。
                        .padding(.bottom, 12)
                    // 動画スクロール領域を閉じます。
                    }
                // 動画有無の条件を閉じます。
                }
            // 結果の縦並びを閉じます。
            }
        // まだ写真を選んでいない場合は次の操作を案内します。
        } else {
            // 顔写真を選択する操作を画像つきで案内します。
            ContentUnavailableView("顔写真を選択してください", systemImage: "person.crop.square", description: Text("左側の顔写真を選ぶと、その人物が映る動画と検出時刻が表示されます。"))
        // 選択状態による表示分岐を閉じます。
        }
    // 結果領域の定義を閉じます。
    }
    // 一つの動画とその中の検出時刻をカードにまとめます。
    private func videoResult(_ video: VideoRecord, faces: [FaceRecord]) -> some View {
        // 動画名と時刻ボタンを縦に並べます。
        VStack(alignment: .leading, spacing: 12) {
            // 動画のファイル名を見出しとして表示します。
            Text(video.name).font(.headline)
            // 検出顔の時刻をボタンのグリッドに並べます。
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 118), spacing: 8)], alignment: .leading, spacing: 8) {
                // この動画で見つかった対象人物の顔だけを選びます。
                ForEach(Array(faces.prefix(visibleFaceCounts[video.id] ?? 60))) { face in
                    // 顔写真と時刻を押すとその位置から動画を開きます。
                    Button {
                        // 再生対象の顔、動画、検出時刻を保存します。
                        selectedMoment = PersonMoment(id: face.id, video: video, second: face.second)
                    // 時刻ボタンの操作を閉じます。
                    } label: {
                        // 顔写真と時刻を横に並べて分かりやすくします。
                        HStack(spacing: 8) {
                            // 顔画像を非同期キャッシュから読み込み、丸く切り抜いて表示します。
                            CachedThumbnail(path: face.thumbnailPath, size: 36, placeholder: "person.crop.circle").clipShape(Circle())
                            // 顔を見つけた時刻を分:秒で表示します。
                            Text(timeLabel(face.second)).monospacedDigit()
                        // 写真と時刻の横並びを閉じます。
                        }
                    // ボタンの内容を閉じます。
                    }
                    // 時刻の移動操作に Liquid Glass を使います。
                    .buttonStyle(.glass)
                // 検出時刻の繰り返しを閉じます。
                }
            // 時刻ボタンのグリッドを閉じます。
            }
            // まだ表示していない検出時刻がある場合だけ追加表示の操作を出します。
            if (visibleFaceCounts[video.id] ?? 60) < faces.count {
                // 押された動画だけ 60 件ぶん表示数を増やします。
                Button("さらに 60 件表示") { visibleFaceCounts[video.id] = min((visibleFaceCounts[video.id] ?? 60) + 60, faces.count) }
                    // 追加表示はカード内の補助操作として示します。
                    .buttonStyle(.plain)
            // 追加表示ボタンの条件を閉じます。
            }
        // 動画カードの内容を閉じます。
        }
        // 動画カード内に余白を設けます。
        .padding(16)
        // 動画名と検出時刻を読みやすい背景に置きます。
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 16))
    // 動画結果カードの定義を閉じます。
    }
    // 現在選択している人物レコードを返します。
    private var selectedPerson: PersonRecord? {
        // 選択 ID に一致する人物をモデルから探します。
        model.people.first { $0.id == selectedPersonID }
    // 選択人物計算の定義を閉じます。
    }
    // 選択したグループまたは未分類の顔一覧を時刻順で返します。
    private var selectedFaces: [FaceRecord] {
        // 除外顔を外し、人物 ID で絞り込んで並べ替えます。
        model.faces.filter { !$0.excluded && (showsUnclassified ? $0.personID == nil : $0.personID == selectedPersonID) }.sorted { $0.second < $1.second }
    // 選択顔一覧計算の定義を閉じます。
    }
    // 秒数を分と秒の読みやすい文字列に変換します。
    private func timeLabel(_ second: Double) -> String {
        // 負の秒数を避け、整数秒に丸めます。
        let value = max(0, Int(second))
        // 分と二桁の秒を組み合わせて返します。
        return String(format: "%d:%02d", value / 60, value % 60)
    // 時刻文字列関数の定義を閉じます。
    }
    // 人物の代表顔写真を角丸タイルで表示します。
    @ViewBuilder private func personPortrait(_ person: PersonRecord, size: CGFloat) -> some View {
        // 代表顔写真をバックグラウンドで縮小読み込みし、タイル内に表示します。
        CachedThumbnail(path: person.thumbnailPath, size: size, placeholder: "person.crop.square.fill").clipShape(RoundedRectangle(cornerRadius: 14))
    // 顔写真表示関数の定義を閉じます。
    }
// PeopleView の定義を閉じます。
}
