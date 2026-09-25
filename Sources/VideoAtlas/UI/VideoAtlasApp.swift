// アプリの画面を作るためにSwiftUIを読み込みます。
import SwiftUI
// 再生画面で動画を表示するためにAVKitを読み込みます。
import AVKit
// macOSのサムネイル画像型を使うためにAppKitを読み込みます。
import AppKit

// VideoAtlasの起動地点と、アプリ全体で共有する状態を定義します。
@main
// SwiftUIアプリであることを示します。
struct VideoAtlasApp: App {
    // 画面をまたいで使うライブラリ状態を保持します。
    @State private var model = AppModel()
    // アプリが開いたときに表示するウィンドウを作ります。
    var body: some Scene {
        // macOS標準のウィンドウを定義します。
        WindowGroup {
            // ライブラリ画面に共有モデルを渡します。
            LibraryView(model: model)
                // 初期サイズを見やすい大きさにします。
                .frame(minWidth: 900, minHeight: 600)
        // ここで直前の処理の範囲を閉じます。
        }
        // ウィンドウ定義の終わりです。
    }
    // アプリ定義の終わりです。
}

// ライブラリ全体のナビゲーションと一覧を表示します。
struct LibraryView: View {
    // 親から受け取った共有モデルを読み書きします。
    var model: AppModel
    // 選択中のサイドバー項目を保持します。
    @State private var selection = LibrarySection.all
    // ファイル選択画面を表示するかを保持します。
    @State private var isChoosingFolder = false
    // 全インデックス削除確認の表示状態を保持します。
    @State private var isConfirmingClear = false
    // 画面本体を組み立てます。
    var body: some View {
        // macOSらしいサイドバー付きレイアウトを作ります。
        NavigationSplitView {
            // 左側にライブラリの移動先を並べます。
            List(selection: $selection) {
                // 全動画を開く項目を作ります。
                Label("すべての動画", systemImage: "film.stack")
                    // 選択値を一覧画面の種別に結び付けます。
                    .tag(LibrarySection.all)
                // 人物一覧を開く項目を作ります。
                Label("人物", systemImage: "person.2")
                    // 選択値を人物画面の種別に結び付けます。
                    .tag(LibrarySection.people)
                // 登録フォルダの見出しと項目を表示します。
                Section("ライブラリフォルダ") {
                    // 登録済みフォルダを順番に表示します。
                    ForEach(model.sources) { source in
                        // フォルダ名とパスを表示します。
                        Label(URL(fileURLWithPath: source.path).lastPathComponent, systemImage: "folder")
                            // パスを補助情報として表示します。
                            .help(source.path)
                            // フォルダを選ぶと、その配下の動画だけを表示します。
                            .tag(LibrarySection.folder(source.id))
                    // ここで直前の処理の範囲を閉じます。
                    }
                    // フォルダ登録ボタンを表示します。
                    Button { isChoosingFolder = true } label: {
                        // ボタンの意味を明示します。
                        Label("フォルダを追加…", systemImage: "folder.badge.plus")
                    // ここで直前の処理の範囲を閉じます。
                    }
                    // ボタンの既定見た目を控えめにします。
                    .buttonStyle(.plain)
                // ここで直前の処理の範囲を閉じます。
                }
                // サイドバー下部に危険操作を置きます。
                Section {
                    // インデックス全消去を確認付きで開始します。
                    Button(role: .destructive) { isConfirmingClear = true } label: {
                        // 削除対象が分かる文言を表示します。
                        Label("インデックスを消去…", systemImage: "trash")
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
                // サイドバー項目の定義を閉じます。
            }
            // サイドバーの見た目を指定します。
            .listStyle(.sidebar)
            // サイドバー上部にアプリ名と操作を配置します。
            .safeAreaInset(edge: .top) {
                // 見出しとフォルダ追加ボタンを横並びにします。
                HStack {
                    // アプリ名を表示します。
                    Text("VideoAtlas").font(.title2.bold())
                    // 余白を使って操作を右寄せします。
                    Spacer()
                    // フォルダ登録画面を開きます。
                    Button { isChoosingFolder = true } label: {
                        // アイコンだけの追加ボタンを作ります。
                        Image(systemName: "plus")
                    // ここで直前の処理の範囲を閉じます。
                    }
                    // 操作名をアクセシビリティ情報に設定します。
                    .help("動画フォルダを追加")
                // ここで直前の処理の範囲を閉じます。
                }
                // 見出しの余白を設定します。
                .padding()
            // ここで直前の処理の範囲を閉じます。
            }
            // サイドバー幅を指定します。
            .navigationSplitViewColumnWidth(min: 220, ideal: 250)
        // ここで直前の処理の範囲を閉じます。
        } detail: {
            // 選択に応じたメイン画面を表示します。
            Group {
                // 人物選択時は人物管理画面を開きます。
                if selection == .people {
                    // 人物一覧と編集操作を表示します。
                    PeopleView(model: model)
                // フォルダ選択時はそのフォルダの動画を表示します。
                } else if case .folder(let sourceID) = selection {
                    // フォルダ ID で絞った動画一覧を表示します。
                    VideoGridView(model: model, sourceID: sourceID)
                // 全動画選択時はすべての動画を表示します。
                } else {
                    // すべての動画カードを表示します。
                    VideoGridView(model: model, sourceID: nil)
                // ここで直前の処理の範囲を閉じます。
                }
                // 画面切替の条件を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // ウィンドウ背景をガラス質感のマテリアルにします。
        .background(.ultraThinMaterial)
        // フォルダ選択をシステムのファイル選択UIで行います。
        .fileImporter(isPresented: $isChoosingFolder, allowedContentTypes: [.folder], allowsMultipleSelection: false) { result in
            // 選択結果が成功したときだけ登録します。
            if case .success(let urls) = result, let url = urls.first {
                // 選択フォルダをスキャン対象に加えます。
                model.addFolder(url: url)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 削除前に対象と影響を確認するダイアログを表示します。
        .confirmationDialog("インデックスを消去しますか？", isPresented: $isConfirmingClear, titleVisibility: .visible) {
            // 確認された場合にモデルのインデックスを消します。
            Button("インデックスを消去", role: .destructive) { model.clearIndex() }
            // 取り消し操作を用意します。
            Button("キャンセル", role: .cancel) {}
        // ここで直前の処理の範囲を閉じます。
        } message: {
            // 消去されるデータを説明します。
            Text("登録した動画情報と人物情報が削除されます。元の動画ファイルは削除されません。")
        // ここで直前の処理の範囲を閉じます。
        }
        // 画面が表示されたときライブラリ情報を更新します。
        .task { model.refreshLibrary() }
        // 一覧画面から届いたフォルダ選択要求を受け取ります。
        .onReceive(NotificationCenter.default.publisher(for: .requestFolderPicker)) { _ in isChoosingFolder = true }
    // ここで直前の処理の範囲を閉じます。
    }
    // ライブラリ画面の定義を閉じます。
}

// サイドバーで選ぶ画面の種類を表します。
private enum LibrarySection: Hashable {
    // すべての動画一覧を表します。
    case all
    // 人物管理画面を表します。
    case people
    // 登録フォルダごとの動画一覧を表します。
    case folder(UUID)
// ここで直前の処理の範囲を閉じます。
}

// 動画をカード状に並べ、検索と解析操作を提供します。
struct VideoGridView: View {
    // 共有モデルを読み書きします。
    var model: AppModel
    // フォルダを選んだ場合だけ、そのフォルダの ID を保持します。
    let sourceID: UUID?
    // 検索入力を保持します。
    @State private var query = ""
    // 再生対象を保持します。
    @State private var selectedVideo: VideoRecord?
    // 動画一覧の本体を作ります。
    var body: some View {
        // 縦方向にタイトル、進捗、カード一覧を並べます。
        VStack(alignment: .leading, spacing: 18) {
            // タイトルと検索欄を横並びにします。
            HStack {
                // 画面見出しを表示します。
                Text(sourceTitle).font(.largeTitle.bold())
                // 残り幅を検索欄の前に置きます。
                Spacer()
                // 動画名とパスを検索できます。
                TextField("動画を検索", text: $query).textFieldStyle(.roundedBorder).frame(width: 240)
                // スキャン中は一時停止ボタンを表示します。
                if model.isScanning {
                    // 現在の解析を一時停止します。
                    Button("一時停止") { model.pauseScanning() }
                // 待機中は再開ボタンを表示します。
                } else {
                    // フォルダの新規・変更動画を確認し、停止中の解析も再開します。
                    Button("フォルダを更新") { model.refreshLibrary() }.buttonStyle(.glassProminent)
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
            // 進捗と現在の状態を表示します。
            if model.isScanning || !model.statusText.isEmpty {
                // 進捗状況をカードにまとめます。
                VStack(alignment: .leading, spacing: 8) {
                    // 状態メッセージを表示します。
                    Text(model.statusText).font(.callout).foregroundStyle(.secondary)
                    // 解析率を視覚的なバーで表示します。
                    ProgressView(value: model.progress)
                // ここで直前の処理の範囲を閉じます。
                }
                // 進捗文字を読みやすい標準マテリアル上に置きます。
                .padding().background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            // ここで直前の処理の範囲を閉じます。
            }
            // 検索条件に合う動画をグリッド表示します。
            ScrollView {
                // 利用可能な幅に合わせてカード列数を調整します。
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 16)], spacing: 16) {
                    // 一致した動画をカードにします。
                    ForEach(filteredVideos) { video in
                        // 再生できる動画カードを表示します。
                        VideoCard(video: video)
                            // カードを押した動画を再生対象にします。
                            .onTapGesture { selectedVideo = video }
                            // カードをドロップ先にしてフォルダ追加を支援します。
                            .dropDestination(for: URL.self) { urls, _ in addDroppedFolders(urls); return true }
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
                // グリッドの周囲に余白を作ります。
                .padding(.bottom, 20)
            // ここで直前の処理の範囲を閉じます。
            }
            // グリッド領域いっぱいに伸ばします。
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // 空のライブラリには次の操作を案内します。
            .overlay {
                // 動画が一件もないとき案内を表示します。
                if filteredVideos.isEmpty {
                    // 説明とフォルダ追加ボタンを表示します。
                    ContentUnavailableView {
                        // 空状態のアイコンを指定します。
                        Label("動画がありません", systemImage: "film")
                    // ここで直前の処理の範囲を閉じます。
                    } description: {
                        // 動画追加方法を説明します。
                        Text("フォルダを追加するか、動画フォルダをここへドラッグしてください。")
                    // ここで直前の処理の範囲を閉じます。
                    } actions: {
                        // フォルダ選択のための通知を親へ伝えます。
                        Button("フォルダを追加") { NotificationCenter.default.post(name: .requestFolderPicker, object: nil) }
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 画面に余白を設定します。
        .padding(24)
        // 選んだ動画をプレイヤーシートで開きます。
        .sheet(item: $selectedVideo) { video in
            // 動画再生と人物検出結果を表示します。
            VideoPlayerView(model: model, video: video)
        // ここで直前の処理の範囲を閉じます。
        }
        // ドロップされたフォルダをライブラリへ登録します。
        .dropDestination(for: URL.self) { urls, _ in addDroppedFolders(urls); return true }
    // ここで直前の処理の範囲を閉じます。
    }
    // 検索入力で絞り込んだ動画を返します。
    private var filteredVideos: [VideoRecord] {
        // 小文字化して大文字小文字を問わない検索にします。
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        // 空欄なら全件を返し、それ以外なら名前とパスを照合します。
        return model.videos.filter { (sourceID == nil || $0.sourceID == sourceID) && (needle.isEmpty || $0.name.lowercased().contains(needle) || $0.path.lowercased().contains(needle)) }
    // ここで直前の処理の範囲を閉じます。
    }
    // 選択したフォルダ名または全動画の見出しを返します。
    private var sourceTitle: String {
        // フォルダ ID と登録情報があればフォルダ名を使います。
        guard let sourceID, let source = model.sources.first(where: { $0.id == sourceID }) else { return "すべての動画" }
        // ファイルパスの最後の部分を読みやすい名前として返します。
        return URL(fileURLWithPath: source.path).lastPathComponent
    // ここで直前の処理の範囲を閉じます。
    }
    // ドラッグされたフォルダURLを追加します。
    private func addDroppedFolders(_ urls: [URL]) {
        // フォルダだけをモデルに登録します。
        for url in urls where (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true { model.addFolder(url: url) }
    // ここで直前の処理の範囲を閉じます。
    }
    // 動画一覧画面の定義を閉じます。
}

// サムネイル、動画名、状態をまとめたカードを表示します。
private struct VideoCard: View {
    // 表示対象の動画レコードを受け取ります。
    let video: VideoRecord
    // カードの表示を作ります。
    var body: some View {
        // 情報を縦方向に並べます。
        VStack(alignment: .leading, spacing: 10) {
            // サムネイルかプレースホルダーを表示します。
            ZStack {
                // 画像がない場合の暗い背景を表示します。
                Rectangle().fill(.quaternary)
                // 動画の代表画像をバックグラウンドで縮小して枠内に表示します。
                CachedThumbnail(path: video.posterPath, size: 210, placeholder: "play.rectangle").frame(height: 125).clipped()
            // ここで直前の処理の範囲を閉じます。
            }
            // サムネイルの縦横比を揃えます。
            .frame(height: 125).clipShape(RoundedRectangle(cornerRadius: 12))
            // ファイル名を最大2行で表示します。
            Text(video.name).font(.headline).lineLimit(2)
            // 再生時間と解析状態を表示します。
            HStack {
                // 秒数を時分秒形式で表示します。
                Text(durationLabel(video.duration)).font(.caption).foregroundStyle(.secondary)
                // 空間を状態ラベルの前に置きます。
                Spacer()
                // 状態を日本語で表示します。
                Text(stateLabel(video.state)).font(.caption).foregroundStyle(.secondary)
            // ここで直前の処理の範囲を閉じます。
            }
            // 壊れた動画や権限不足の理由をカード内に表示します。
            if let message = video.errorMessage {
                // 長いエラーでもカードが伸びすぎないようにします。
                Text(message).font(.caption).foregroundStyle(.orange).lineLimit(3)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // カード周囲に余白を作ります。
        .padding(12)
        // 動画名が読みやすい不透明なカード背景を使います。
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 18))
        // カードを操作対象として示します。
        .contentShape(RoundedRectangle(cornerRadius: 18))
    // ここで直前の処理の範囲を閉じます。
    }
    // 秒数を読みやすい時間表示へ変換します。
    private func durationLabel(_ seconds: Double) -> String {
        // 時間と分と秒を整数で組み立てます。
        let value = max(0, Int(seconds))
        // 1時間以上は時:分:秒で表示します。
        return value >= 3600 ? String(format: "%d:%02d:%02d", value / 3600, (value / 60) % 60, value % 60) : String(format: "%d:%02d", value / 60, value % 60)
    // ここで直前の処理の範囲を閉じます。
    }
    // 解析状態を日本語の短いラベルへ変換します。
    private func stateLabel(_ state: VideoState) -> String {
        // enumのケースごとに状態を説明します。
        switch state {
        // 未処理を待機中と表示します。
        case .pending: return "待機中"
        // 解析中を表示します。
        case .analyzing: return "解析中"
        // 完了を表示します。
        case .ready: return "完了"
        // 元ファイル不在を表示します。
        case .missing: return "ファイルなし"
        // 失敗を表示します。
        case .failed: return "失敗"
        // 状態変換を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }
    // カード定義を閉じます。
}

// 動画の再生、シーク、検出人物の確認を行います。
struct VideoPlayerView: View {
    // シートを閉じるための SwiftUI 操作を受け取ります。
    @Environment(\.dismiss) private var dismiss
    // アプリ状態を受け取ります。
    var model: AppModel
    // 再生する動画レコードを受け取ります。
    let video: VideoRecord
    // 人物画面の検出時刻から開いたときの再生開始秒数です。
    let initialSecond: Double
    // AVPlayerを画面表示中に保持します。
    @State private var player: AVPlayer
    // 動画プレイヤーを準備します。
    init(model: AppModel, video: VideoRecord, initialSecond: Double = 0) {
        // モデルを保存します。
        self.model = model
        // 動画レコードを保存します。
        self.video = video
        // 押された検出時刻を保存します。
        self.initialSecond = initialSecond
        // ファイルURLから再生機器を作ります。
        _player = State(initialValue: AVPlayer(url: URL(fileURLWithPath: video.path)))
    // ここで直前の処理の範囲を閉じます。
    }
    // 再生画面を組み立てます。
    var body: some View {
        // プレイヤーと検出結果を上下に並べます。
        VStack(alignment: .leading, spacing: 14) {
            // ファイル名と詳細再解析操作を表示します。
            HStack {
                // 動画名を表示します。
                Text(video.name).font(.title2.bold()).lineLimit(1)
                // 空き幅で右端へボタンを寄せます。
                Spacer()
                // 現在の動画を詳細モードで再解析します。
                Button("詳しく再解析") { model.rescan(videoID: video.id, detailed: true) }.buttonStyle(.glass)
                // 再生シートを閉じる操作を表示します。
                Button("閉じる") { dismiss() }.buttonStyle(.glass)
            // ここで直前の処理の範囲を閉じます。
            }
            // macOS 標準の再生ビューを表示します。
            NativeVideoPlayer(player: player).frame(minHeight: 320)
            // 検出した人物サムネイルの見出しを表示します。
            Text("この動画に登場する人物").font(.headline)
            // 解析中は編集を止める必要があるため説明と停止操作を表示します。
            if model.isScanning {
                // 一時停止後に顔の修正を行えることを案内します。
                HStack(spacing: 10) {
                    // 解析中は顔の割当変更などを使えない理由を説明します。
                    Text("顔の修正は解析を一時停止してから行えます").font(.caption).foregroundStyle(.secondary)
                    // 現在の動画を保存してから解析を停止します。
                    Button("一時停止") { model.pauseScanning() }
                // 注意文と停止ボタンの横並びを閉じます。
                }
            // 解析中に限った案内表示の条件を閉じます。
            }
            // 検出顔を横スクロールで並べます。
            ScrollView(.horizontal) {
                // 対象動画の顔だけを表示します。
                LazyHStack(spacing: 12) {
                    // 除外されていない顔を選びます。
                    ForEach(model.faces.filter { $0.videoID == video.id && !$0.excluded }) { face in
                        // 顔サムネイルと時刻を操作カードとして表示します。
                        FaceMomentCard(model: model, face: face, player: player)
                    // ここで直前の処理の範囲を閉じます。
                    }
                // ここで直前の処理の範囲を閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            }
            // 顔が見つからない場合の状態を説明します。
            if model.faces.filter({ $0.videoID == video.id && !$0.excluded }).isEmpty {
                // 解析待ちか未検出かを断定せず案内します。
                Text(video.state == .ready ? "この動画では顔が検出されませんでした。必要なら詳しく再解析できます。" : "人物の検出結果はまだありません。解析完了後にここへ表示されます。").foregroundStyle(.secondary)
            // ここで直前の処理の範囲を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
        // 再生画面に余白を付けます。
        .padding(20)
        // シート内でも十分な領域を確保します。
        .frame(minWidth: 760, minHeight: 560)
        // 表示時に自動再生しないよう一時停止状態にします。
        .onAppear {
            // 自動再生は行わず、指定された検出時刻まで移動します。
            player.pause()
            // 人物画面から開いた場合だけ、その顔の時刻を表示します。
            // 指定時刻からずれにくくするため前後の許容幅をゼロにします。
            if initialSecond > 0 {
                // 再生位置を要求された時刻へ厳密に移動します。
                player.seek(to: CMTime(seconds: initialSecond, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
            // 初期時刻への移動条件を閉じます。
            }
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }
    // プレイヤー画面の定義を閉じます。
}

// AVPlayerView を SwiftUI の画面内で使えるように橋渡しします。
private struct NativeVideoPlayer: NSViewRepresentable {
    // 再生する動画を管理する AVPlayer を受け取ります。
    let player: AVPlayer
    // SwiftUI が最初に表示する macOS の再生ビューを作ります。
    func makeNSView(context: Context) -> AVPlayerView {
        // 再生ボタンやシーク操作を備えたビューを作ります。
        let view = AVPlayerView()
        // 動画をビューへ接続します。
        view.player = player
        // 標準の再生操作を表示します。
        view.controlsStyle = .floating
        // 完成したビューを SwiftUI へ渡します。
        return view
    // ここで直前の処理の範囲を閉じます。
    }
    // SwiftUI が再描画するときに再生対象を最新にします。
    func updateNSView(_ view: AVPlayerView, context: Context) {
        // 現在の AVPlayer を再生ビューへ結び付けます。
        view.player = player
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// 検出顔を押すと対応する動画時刻へ移動するカードです。
private struct FaceMomentCard: View {
    // モデル操作に使う共有状態を受け取ります。
    var model: AppModel
    // 表示する顔レコードを受け取ります。
    let face: FaceRecord
    // シーク対象のプレイヤーを受け取ります。
    let player: AVPlayer
    // 顔カードを作ります。
    var body: some View {
        // 時刻とサムネイルを縦にまとめます。
        VStack(alignment: .leading, spacing: 6) {
            // カードを押すと検出時刻へシークします。
            Button {
                // 検出時刻へ前後の許容幅なしで移動します。
                player.seek(to: CMTime(seconds: face.second, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
            // シークボタンの操作を閉じます。
            } label: {
                // 顔画像をバックグラウンドで縮小し、角丸カードへ表示します。
                CachedThumbnail(path: face.thumbnailPath, size: 92, placeholder: "person.crop.square").clipShape(RoundedRectangle(cornerRadius: 12))
            // ここで直前の処理の範囲を閉じます。
            }
            // 時刻への移動操作をカードらしく表示します。
            .buttonStyle(.plain)
            // 検出秒を時刻ラベルとして表示します。
            Text(timeLabel(face.second)).font(.caption.monospacedDigit())
            // 顔単位の管理操作をメニューにまとめます。
            Menu {
                // 顔を人物分類から外す操作を用意します。
                Button("この顔を除外", systemImage: "eye.slash") { model.excludeFace(faceID: face.id) }
                // この顔だけを独立したグループへ分ける操作を用意します。
                Button("別の顔グループに分ける", systemImage: "person.crop.circle.badge.plus") { model.splitFace(faceID: face.id) }
                // 誤分類された顔の移動先を顔写真で選べるメニューを作ります。
                Menu("顔写真のグループへ移動") {
                    // 登録済みグループを顔写真の選択肢として並べます。
                    ForEach(model.people) { person in
                        // 写真を見て移動先を選び、この顔をそのグループへ割り当てます。
                        Button {
                            // 選択した顔写真の人物グループへ割り当てます。
                            model.assignFace(faceID: face.id, to: person.id)
                        // 顔写真つき選択項目の操作を閉じます。
                        } label: {
                            // 人物名を使わず顔写真と説明だけで移動先を示します。
                            HStack {
                                // 移動先の代表写真を非同期キャッシュから小さく読み込みます。
                                CachedThumbnail(path: person.thumbnailPath, size: 26, placeholder: "person.crop.circle").clipShape(Circle())
                                // 項目の動作を短い日本語で説明します。
                                Text("この顔写真のグループへ移動")
                            // 写真と説明文の横並びを閉じます。
                            }
                        // メニュー項目の内容を閉じます。
                        }
                    // 顔写真候補の繰り返しを閉じます。
                    }
                    // 自動の人物割当を外して未分類へ戻す操作を用意します。
                    Button("未分類の顔に戻す") { model.assignFace(faceID: face.id, to: nil) }
                // 顔写真で選ぶ移動先メニューを閉じます。
                }
            // ここで直前の処理の範囲を閉じます。
            } label: {
                // 操作メニューを開く小さなボタンを表示します。
                Label("顔の操作", systemImage: "ellipsis.circle").labelStyle(.titleAndIcon).font(.caption)
            // ここで直前の処理の範囲を閉じます。
            }
            // 解析中は除外・分割・割当の顔編集メニューを使えないようにします。
            .disabled(model.isScanning)
        // ここで直前の処理の範囲を閉じます。
        }
        // 顔の画像を読みやすい不透明な背景に置きます。
        .padding(8).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14))
    // ここで直前の処理の範囲を閉じます。
    }
    // 時刻を分:秒形式にします。
    private func timeLabel(_ second: Double) -> String {
        // 0未満を防いで整数秒へ変換します。
        let value = max(0, Int(second))
        // 分と秒を2桁の秒で表示します。
        return String(format: "%d:%02d", value / 60, value % 60)
    // ここで直前の処理の範囲を閉じます。
    }
    // 顔カード定義を閉じます。
}

// 画面間でフォルダ選択を要求する通知名を定義します。
private extension Notification.Name {
    // フォルダ選択通知を識別する名前です。
    static let requestFolderPicker = Notification.Name("VideoAtlasRequestFolderPicker")
// ここで直前の処理の範囲を閉じます。
}
