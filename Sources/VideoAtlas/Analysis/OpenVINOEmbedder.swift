// OpenVINO の共有ライブラリを実行時に読み込むため、動的リンクの標準 API を使います。
import Darwin
// モデルファイルの場所とエラー文を扱うため、Foundation を読み込みます。
import Foundation

// 解析機能で表示するエラーを、日本語の説明付きで定義します。
enum AnalysisError: LocalizedError {
    // 動画が開けない場合のエラーを表します。
    case invalidVideo(String)
    // モデルや OpenVINO が見つからない場合のエラーを表します。
    case unavailable(String)
    // 推論などの処理に失敗した場合のエラーを表します。
    case processing(String)
    // 利用者に表示する説明文を返します。
    var errorDescription: String? {
        // エラーの種類ごとに保持していた説明文を返します。
        switch self {
        // 動画の説明文を返します。
        case .invalidVideo(let message): return message
        // 環境不足の説明文を返します。
        case .unavailable(let message): return message
        // 処理失敗の説明文を返します。
        case .processing(let message): return message
        // ここで直前の処理の範囲を閉じます。
        }
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}

// OpenVINO の C API を Swift から型付き関数として呼び出します。
final class OpenVINOEmbedder: @unchecked Sendable {
    // OpenVINO の成功値は整数の 0 です。
    private static let success: Int32 = 0
    // C API が保持するライブラリを、オブジェクトの存続中は閉じずに保管します。
    private let library: UnsafeMutableRawPointer
    // 読み込んだ Core を保持します。
    private var core: UnsafeMutableRawPointer?
    // コンパイル済みモデルを保持します。
    private var compiled: UnsafeMutableRawPointer?
    // 推論要求を保持します。
    private var request: UnsafeMutableRawPointer?
    // 複数の解析が同じ推論要求を同時に使わないようにします。
    private let lock = NSLock()
    // Core を作る C 関数の型を宣言します。
    private typealias CreateCore = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // モデルを読む C 関数の型を宣言します。
    private typealias ReadModel = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, UnsafePointer<CChar>?, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // モデルをコンパイルする C 関数の型を宣言します。
    private typealias CompileModel = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Int, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // 推論要求を作る C 関数の型を宣言します。
    private typealias CreateRequest = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // 入出力テンソルを取得する C 関数の型を宣言します。
    private typealias GetTensor = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // テンソルのバイト数を得る C 関数の型を宣言します。
    private typealias TensorByteSize = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?) -> Int32
    // テンソルのメモリへのポインターを得る C 関数の型を宣言します。
    private typealias TensorData = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32
    // 推論を実行する C 関数の型を宣言します。
    private typealias Infer = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    // C API のオブジェクトを解放する関数の型を宣言します。
    private typealias FreeObject = @convention(c) (UnsafeMutableRawPointer?) -> Void
    // C API が返した最後の詳細エラーを読む関数の型を宣言します。
    private typealias LastError = @convention(c) () -> UnsafePointer<CChar>?
    // 動的に読み込んだ各関数を保存します。
    private let createCore: CreateCore
    // モデル読込関数を保存します。
    private let readModel: ReadModel
    // モデルコンパイル関数を保存します。
    private let compileModel: CompileModel
    // 推論要求作成関数を保存します。
    private let createRequest: CreateRequest
    // 入力テンソル取得関数を保存します。
    private let getInput: GetTensor
    // 出力テンソル取得関数を保存します。
    private let getOutput: GetTensor
    // テンソル長取得関数を保存します。
    private let tensorByteSize: TensorByteSize
    // テンソルデータ取得関数を保存します。
    private let tensorData: TensorData
    // 推論関数を保存します。
    private let infer: Infer
    // Core 解放関数を保存します。
    private let freeCore: FreeObject
    // モデル解放関数を保存します。
    private let freeModel: FreeObject
    // コンパイル済みモデル解放関数を保存します。
    private let freeCompiled: FreeObject
    // 推論要求解放関数を保存します。
    private let freeRequest: FreeObject
    // テンソル解放関数を保存します。
    private let freeTensor: FreeObject
    // 最後の OpenVINO エラーを表示する関数を保存します。
    private let lastError: LastError

    // アプリ付属モデルと実行時ライブラリを見つけて推論器を準備します。
    init() throws {
        // Bundle.module は Swift Package に登録されたリソースを指します。
        let bundle = Bundle.module
        // リソースの Models ディレクトリから XML モデルを探します。
        guard let modelURL = bundle.url(forResource: "face-reidentification-retail-0095", withExtension: "xml", subdirectory: "Models"),
              // XML と同じ名前の重みファイルがあることを確かめます。
              let weightsURL = bundle.url(forResource: "face-reidentification-retail-0095", withExtension: "bin", subdirectory: "Models") else {
            // モデルがなければ不足しているファイル名を具体的に伝えます。
            throw AnalysisError.unavailable("顔識別モデルの XML/BIN がアプリの Models リソースにありません。")
        // ここで直前の処理の範囲を閉じます。
        }
        // 環境変数で指定された開発用ディレクトリを読み取ります。
        let environmentDirectory = ProcessInfo.processInfo.environment["VIDEOATLAS_OPENVINO_DIR"]
        // アプリ内、指定先、Homebrew の順にライブラリ候補を組み立てます。
        let candidates = [bundle.resourceURL?.appendingPathComponent("OpenVINO/libopenvino_c.dylib").path,
                          // 指定された場所にある OpenVINO 共有ライブラリを候補へ加えます。
                          environmentDirectory.map { URL(fileURLWithPath: $0).appendingPathComponent("libopenvino_c.dylib").path },
                          // 指定された場所にある OpenVINO 共有ライブラリを候補へ加えます。
                          environmentDirectory.map { URL(fileURLWithPath: $0).appendingPathComponent("runtime/lib/intel64/libopenvino_c.dylib").path },
                          // 標準の場所にある OpenVINO 共有ライブラリを候補へ加えます。
                          "/opt/homebrew/lib/libopenvino_c.dylib",
                          // 標準の場所にある OpenVINO 共有ライブラリを候補へ加えます。
                          "/usr/local/lib/libopenvino_c.dylib"].compactMap { $0 }
        // 最初に開けたライブラリのハンドルを保持します。
        guard let handle = candidates.lazy.compactMap({ dlopen($0, RTLD_NOW | RTLD_LOCAL) }).first else {
            // 開けない場合はアプリ付属ディレクトリと環境変数を案内します。
            throw AnalysisError.unavailable("OpenVINO C Runtime を開けません。Bundle.module/OpenVINO または VIDEOATLAS_OPENVINO_DIR に libopenvino_c.dylib と依存ライブラリを配置してください。")
        // ここで直前の処理の範囲を閉じます。
        }
        // 以降の関数解決と初期化で使うハンドルを保存します。
        library = handle
        // 公開 C API の名前から Core 作成関数を解決します。
        createCore = try Self.symbol(handle, "ov_core_create")
        // 公開 C API の名前からモデル読込関数を解決します。
        readModel = try Self.symbol(handle, "ov_core_read_model")
        // 公開 C API の名前からモデルコンパイル関数を解決します。
        compileModel = try Self.symbol(handle, "ov_core_compile_model")
        // 公開 C API の名前から推論要求作成関数を解決します。
        createRequest = try Self.symbol(handle, "ov_compiled_model_create_infer_request")
        // 公開 C API の名前から入力取得関数を解決します。
        getInput = try Self.symbol(handle, "ov_infer_request_get_input_tensor")
        // 公開 C API の名前から出力取得関数を解決します。
        getOutput = try Self.symbol(handle, "ov_infer_request_get_output_tensor")
        // 公開 C API の名前からテンソル長取得関数を解決します。
        tensorByteSize = try Self.symbol(handle, "ov_tensor_get_byte_size")
        // 公開 C API の名前からテンソルデータ取得関数を解決します。
        tensorData = try Self.symbol(handle, "ov_tensor_data")
        // 公開 C API の名前から推論関数を解決します。
        infer = try Self.symbol(handle, "ov_infer_request_infer")
        // 公開 C API の名前から Core 解放関数を解決します。
        freeCore = try Self.symbol(handle, "ov_core_free")
        // 公開 C API の名前からモデル解放関数を解決します。
        freeModel = try Self.symbol(handle, "ov_model_free")
        // 公開 C API の名前からコンパイル済みモデル解放関数を解決します。
        freeCompiled = try Self.symbol(handle, "ov_compiled_model_free")
        // 公開 C API の名前から推論要求解放関数を解決します。
        freeRequest = try Self.symbol(handle, "ov_infer_request_free")
        // 公開 C API の名前からテンソル解放関数を解決します。
        freeTensor = try Self.symbol(handle, "ov_tensor_free")
        // 公開 C API の名前から詳細エラー取得関数を解決します。
        lastError = try Self.symbol(handle, "ov_get_last_err_msg")
        // Core の出力先を初期化します。
        var createdCore: UnsafeMutableRawPointer?
        // Core の作成結果を確認します。
        guard createCore(&createdCore) == Self.success, let createdCore else {
            // Core を作れなかった詳細を報告します。
            throw AnalysisError.unavailable("OpenVINO Core を作成できません。CPU プラグインと依存ライブラリを確認してください。")
        // ここで直前の処理の範囲を閉じます。
        }
        // 後から解放できるように Core を保存します。
        core = createdCore
        // モデルの出力先を初期化します。
        var model: UnsafeMutableRawPointer?
        // XML と BIN の実際のパスを C 文字列にしてモデルを読み込みます。
        let readStatus = modelURL.path.withCString { xml in weightsURL.path.withCString { bin in readModel(createdCore, xml, bin, &model) } }
        // 読込失敗なら OpenVINO の説明を含むエラーを返します。
        try check(readStatus, "モデルの読み込み")
        // モデルのポインターが存在することを確認します。
        guard let model else { throw AnalysisError.processing("モデルの読み込みで空の結果が返りました。") }
        // モデルはコンパイル後に不要なので必ず解放します。
        defer { freeModel(model) }
        // コンパイル済みモデルの出力先を初期化します。
        var createdCompiled: UnsafeMutableRawPointer?
        // CPU を指定してモデルをコンパイルします。
        let compileStatus = "CPU".withCString { compileModel(createdCore, model, $0, 0, &createdCompiled) }
        // コンパイル失敗ならプラグインなどの詳細を返します。
        try check(compileStatus, "CPU 用モデルのコンパイル")
        // コンパイル済みモデルが存在することを確認します。
        guard let createdCompiled else { throw AnalysisError.processing("コンパイル済みモデルが空です。") }
        // オブジェクト解放までコンパイル済みモデルを保存します。
        compiled = createdCompiled
        // 推論要求の出力先を初期化します。
        var createdRequest: UnsafeMutableRawPointer?
        // 入出力テンソルを持つ推論要求を作ります。
        try check(createRequest(createdCompiled, &createdRequest), "推論要求の作成")
        // 推論要求が存在することを確認します。
        guard let createdRequest else { throw AnalysisError.processing("推論要求が空です。") }
        // 後から再利用できるように推論要求を保存します。
        request = createdRequest
    // ここで直前の処理の範囲を閉じます。
    }

    // Swift の関数型に変換して C API の公開シンボルを返します。
    private static func symbol<T>(_ handle: UnsafeMutableRawPointer, _ name: String) throws -> T {
        // dlsym で目的のシンボルを探します。
        guard let address = dlsym(handle, name) else { throw AnalysisError.unavailable("OpenVINO に \(name) がありません。対応する C Runtime を確認してください。") }
        // ABI が対応する C 関数ポインターとして読み替えます。
        return unsafeBitCast(address, to: T.self)
    // ここで直前の処理の範囲を閉じます。
    }

    // OpenVINO が返した状態と詳細説明を合わせて検査します。
    private func check(_ status: Int32, _ step: String) throws {
        // 正常なら何もしません。
        guard status != Self.success else { return }
        // C 文字列の詳細エラーが存在すれば Swift の文字列にします。
        let detail = lastError().map { String(cString: $0) } ?? "詳細不明"
        // 処理段階、状態番号、詳細を併記します。
        throw AnalysisError.processing("OpenVINO の\(step)に失敗しました（状態 \(status)）：\(detail)")
    // ここで直前の処理の範囲を閉じます。
    }

    // 128x128 の BGR 平面配列から 256 次元の特徴ベクトルを返します。
    func embedding(bgr: [Float]) throws -> [Float] {
        // モデルの入力は 3 色 × 128 × 128 個の値です。
        guard bgr.count == 3 * 128 * 128 else { throw AnalysisError.processing("顔画像の入力サイズが 128×128 BGR ではありません。") }
        // 一つの推論要求を安全に再利用するためロックします。
        lock.lock()
        // 正常時も失敗時もロックを解除します。
        defer { lock.unlock() }
        // 初期化済みの推論要求を取り出します。
        guard let request else { throw AnalysisError.processing("OpenVINO の推論要求がありません。") }
        // 入力テンソルの出力先を初期化します。
        var input: UnsafeMutableRawPointer?
        // 入力テンソルを取得します。
        try check(getInput(request, &input), "入力テンソルの取得")
        // 入力テンソルが存在することを確認します。
        guard let input else { throw AnalysisError.processing("入力テンソルが空です。") }
        // 取得した C API のテンソルを処理後に解放します。
        defer { freeTensor(input) }
        // 入力テンソルのバイト数を初期化します。
        var inputBytes = 0
        // テンソルの型と形状が必要な FP32 サイズであることを検査します。
        try check(tensorByteSize(input, &inputBytes), "入力サイズの取得")
        // 異なるモデルを誤って使わないようにサイズを比較します。
        guard inputBytes == bgr.count * MemoryLayout<Float>.size else { throw AnalysisError.processing("モデル入力の形状または型が 1×3×128×128 FP32 ではありません。") }
        // 入力テンソルの書込先を初期化します。
        var inputData: UnsafeMutableRawPointer?
        // 入力テンソルのメモリを取得します。
        try check(tensorData(input, &inputData), "入力メモリの取得")
        // メモリが存在することを確認します。
        guard let inputData else { throw AnalysisError.processing("入力メモリが空です。") }
        // Swift 配列の連続メモリをテンソルにコピーします。
        bgr.withUnsafeBufferPointer { source in if let base = source.baseAddress { inputData.copyMemory(from: base, byteCount: inputBytes) } }
        // CPU で同期推論を実行します。
        try check(infer(request), "顔特徴量の推論")
        // 出力テンソルの出力先を初期化します。
        var output: UnsafeMutableRawPointer?
        // 出力テンソルを取得します。
        try check(getOutput(request, &output), "出力テンソルの取得")
        // 出力テンソルが存在することを確認します。
        guard let output else { throw AnalysisError.processing("出力テンソルが空です。") }
        // 出力テンソルを処理後に解放します。
        defer { freeTensor(output) }
        // 出力テンソルのバイト数を初期化します。
        var outputBytes = 0
        // 出力テンソルの長さを調べます。
        try check(tensorByteSize(output, &outputBytes), "出力サイズの取得")
        // 256 個の FP32 値であることを検査します。
        guard outputBytes == 256 * MemoryLayout<Float>.size else { throw AnalysisError.processing("モデル出力が 256 次元 FP32 ではありません。") }
        // 出力テンソルの読取先を初期化します。
        var outputData: UnsafeMutableRawPointer?
        // 出力テンソルのメモリを取得します。
        try check(tensorData(output, &outputData), "出力メモリの取得")
        // メモリが存在することを確認します。
        guard let outputData else { throw AnalysisError.processing("出力メモリが空です。") }
        // 256 個の浮動小数を Swift 配列としてコピーします。
        return Array(UnsafeBufferPointer(start: outputData.assumingMemoryBound(to: Float.self), count: 256))
    // ここで直前の処理の範囲を閉じます。
    }

    // 使用した C API オブジェクトとライブラリを後片付けします。
    deinit {
        // 推論要求があれば先に解放します。
        if let request { freeRequest(request) }
        // コンパイル済みモデルがあれば解放します。
        if let compiled { freeCompiled(compiled) }
        // Core があれば最後に解放します。
        if let core { freeCore(core) }
        // C API のオブジェクトをすべて解放した後にライブラリを閉じます。
        dlclose(library)
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
