// 動画から指定時刻の画像を取り出すために AVFoundation を読み込みます。
import AVFoundation
// 顔画像の描画と画素処理のために CoreGraphics を読み込みます。
import CoreGraphics
// ファイル、画像データ、数値を扱うために Foundation を読み込みます。
import Foundation
// JPEG のサムネイルを作るために AppKit を読み込みます。
import AppKit
// 顔の検出と目・鼻・口の位置を得るために Vision を読み込みます。
import Vision

// 一つの時刻で見つかった顔と、その画像・特徴量を保持します。
struct AnalyzedFace: Sendable {
    // 動画内で実際に取り出されたフレームの秒数です。
    let second: Double
    // フレーム内で検出した顔の相対位置と大きさです。
    let boundingBox: CGRect
    // 整列済み顔画像の JPEG データです。
    let thumbnailData: Data
    // 顔の比較に使う 256 次元の値です。
    let embedding: [Float]?
// ここで直前の処理の範囲を閉じます。
}

// 動画一件の解析結果と再開に必要な時刻を保持します。
struct VideoAnalysisResult: Sendable {
    // メディアから読み取った動画の総秒数です。
    let duration: Double
    // 最初に取得できたフレームの JPEG データです。
    let posterData: Data?
    // 見つかった顔を時刻順に並べます。
    let faces: [AnalyzedFace]
    // 処理済みフレームの次に読むべき時刻を記録します。
    let completedSecond: Double
    // 利用者の中断指示により途中で終えたかを示します。
    let wasCancelled: Bool
// ここで直前の処理の範囲を閉じます。
}

// 動画のフレーム取得、顔検出、顔特徴量の計算をまとめます。
final class VideoAnalyzer: @unchecked Sendable {
    // 推論器を一つだけ保持して繰り返し利用します。
    private let embedder: OpenVINOEmbedder
    // 通常解析で生成する画像の最長辺を制限します。
    private let maximumDimension: Double
    // モデルを読み込み、失敗したら理由を呼び出し元へ返します。
    init(maximumDimension: Double = 1280) throws {
        // 検証用にゼロを渡した場合だけ元解像度を使います。
        self.maximumDimension = maximumDimension
        // OpenVINO と顔モデルを準備します。
        embedder = try OpenVINOEmbedder()
    // ここで直前の処理の範囲を閉じます。
    }

    // 指定間隔で動画を調べ、中断時には再開位置を返します。
    func analyze(url: URL, interval: Double, startingAt: Double, endingAt: Double? = nil, progress: @escaping @Sendable (Double, Double) -> Void, isCancelled: @escaping @Sendable () -> Bool) async throws -> VideoAnalysisResult {
        // 不正な間隔による無限ループを防ぎます。
        guard interval.isFinite, interval > 0 else { throw AnalysisError.processing("解析間隔は 0 秒より大きい有限値にしてください。") }
        // 再開位置に不正な値が入るのを防ぎます。
        guard startingAt.isFinite, startingAt >= 0 else { throw AnalysisError.processing("再開位置は 0 秒以上の有限値にしてください。") }
        // 画像生成や Vision の重い処理を呼び出し元の実行文脈から切り離します。
        return try await Task.detached(priority: .utility) { [self] in
            // 非同期の動画情報取得を含む実処理を実行します。
            try await analyzeFrames(url: url, interval: interval, startingAt: startingAt, endingAt: endingAt, progress: progress, isCancelled: isCancelled)
        // ここで直前の処理の範囲を閉じます。
        }.value
    // ここで直前の処理の範囲を閉じます。
    }

    // 動画の各要求時刻を巡ってフレームを調べます。
    private func analyzeFrames(url: URL, interval: Double, startingAt: Double, endingAt: Double? = nil, progress: @escaping @Sendable (Double, Double) -> Void, isCancelled: @escaping @Sendable () -> Bool) async throws -> VideoAnalysisResult {
        // URL を AVFoundation の動画アセットとして開きます。
        let asset = AVURLAsset(url: url)
        // 動画の再生時間を非同期に読み取ります。
        let durationTime = try await asset.load(.duration)
        // 時間を秒数に直します。
        let duration = CMTimeGetSeconds(durationTime)
        // 不正な長さや音声だけのファイルを動画として扱わないよう確認します。
        guard duration.isFinite, duration > 0, !(try await asset.loadTracks(withMediaType: .video)).isEmpty else { throw AnalysisError.invalidVideo("有効な映像トラックを読み取れません：\(url.lastPathComponent)") }
        // 指定時刻の静止画像を生成するオブジェクトを作ります。
        let generator = AVAssetImageGenerator(asset: asset)
        // 映像トラックの回転・向きを画像に反映します。
        generator.appliesPreferredTrackTransform = true
        // 詳細解析は元の解像度を保ち、通常解析は画素数を抑えます。
        if interval > 0.5 && maximumDimension > 0 { generator.maximumSize = CGSize(width: maximumDimension, height: maximumDimension) }
        // 今回処理する区間の末尾を決めます。
        let end = min(duration, endingAt ?? duration)
        // 空の区間で再開位置が進まないことを防ぎます。
        guard end > startingAt || startingAt >= duration else { throw AnalysisError.processing("解析区間の終点が開始位置以前です。") }
        // 進捗通知の頻度を毎秒五回以内に抑える時計です。
        var lastNotification = Date.distantPast
        // 指定した時刻の前後 0.25 秒以内の実フレームを許容します。
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.25, preferredTimescale: 600)
        // 前方についても同じ許容範囲を設定します。
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.25, preferredTimescale: 600)
        // 最初に読めたフレームを一覧のポスターとして保管します。
        var poster: Data?
        // 見つかった顔を蓄積します。
        var faces: [AnalyzedFace] = []
        // 再開時に次に読むべき時刻を記録します。
        var completed = min(startingAt, duration)
        // 利用者が終了を指示したか記録します。
        var cancelled = false
        // startingAt は未処理の最初の要求時刻として扱います。
        var requested = startingAt
        // 動画の末尾より前の時刻だけを処理します。
        while requested < end {
            // Swift タスクまたは画面から中断された場合は部分結果を返します。
            if Task.isCancelled || isCancelled() { cancelled = true; break }
            // 長い動画でも毎回有効な時刻を作ります。
            let time = CMTime(seconds: requested, preferredTimescale: 600)
            // フレームを取り出し、実際の画像時刻も受け取ります。
            let frame: (CGImage, CMTime)
            // 壊れたフレームは動画一件のエラーとして扱います。
            do {
                // AVFoundation から実際のフレームを取り出します。
                frame = try await generator.image(at: time)
            // 直前の処理を閉じ、失敗した場合の処理を開始します。
            } catch {
                // 失敗したファイル名と要求秒数を説明します。
                throw AnalysisError.invalidVideo("\(url.lastPathComponent) の \(String(format: "%.2f", requested)) 秒で映像を取り出せません：\(error.localizedDescription)")
            // ここで直前の処理の範囲を閉じます。
            }
            // フレームごとの一時画像と Vision の内部メモリをすぐに解放します。
            try autoreleasepool {
                // 最初の画像を必要なら JPEG に変換します。
                if poster == nil { poster = Self.jpeg(frame.0, quality: 0.75) }
                // AVFoundation が返した実際のフレーム時刻を秒に直します。
                let actualSecond = CMTimeGetSeconds(frame.1)
                // 顔とそのランドマークを検出します。
                let request = VNDetectFaceLandmarksRequest()
                // Vision の画像要求を、向き補正済みのフレームで作成します。
                let handler = VNImageRequestHandler(cgImage: frame.0, options: [:])
                // 一枚の画像に対して顔検出を実行します。
                try handler.perform([request])
                // 検出された各顔を順番に処理します。
                for face in request.results ?? [] {
                    // 64 ピクセル以上で五点を得られた顔だけを整列します。
                    let aligned = Self.alignedFace(frame.0, face: face)
                    // 整列できない顔は元フレームの顔領域を切り出して残します。
                    guard let thumbnail = Self.faceCrop(frame.0, face: face) ?? aligned?.jpeg else { throw AnalysisError.processing("検出した顔の画像を JPEG に変換できません。") }
                    // 整列できた顔だけから比較用の特徴量を求めます。
                    let embedding = try aligned.map { try embedder.embedding(bgr: $0.bgr) }
                    // 実フレーム時刻と顔画像を特徴量の有無にかかわらず保存します。
                    faces.append(AnalyzedFace(second: actualSecond.isFinite ? actualSecond : requested, boundingBox: face.boundingBox, thumbnailData: thumbnail, embedding: embedding))
                // ここで直前の処理の範囲を閉じます。
                }
            // 一枚分の一時メモリを解放します。
            }
            // 次に読むべき時刻を記録し、再開時の重複処理を防ぎます。
            completed = min(requested + interval, end)
            // 画面側へ現在位置と動画の総秒数を通知します。
            if Date().timeIntervalSince(lastNotification) >= 0.2 {
                // 必要な頻度だけ画面へ通知します。
                progress(completed, duration)
                // 今回の通知時刻を保存します。
                lastNotification = Date()
            // 通知間隔の条件を閉じます。
            }
            // 次の要求時刻へ進めます。
            requested += interval
        // ここで直前の処理の範囲を閉じます。
        }
        // 完了時は総秒数まで終えたことを記録します。
        if !cancelled { completed = end; progress(end, duration) }
        // 再開可能な結果を呼び出し元に返します。
        return VideoAnalysisResult(duration: duration, posterData: poster, faces: faces, completedSecond: completed, wasCancelled: cancelled)
    // ここで直前の処理の範囲を閉じます。
    }

    // 五つの顔ランドマークから整列済み 128×128 画像を作ります。
    private static func alignedFace(_ image: CGImage, face: VNFaceObservation) -> (bgr: [Float], jpeg: Data)? {
        // 元画像上の顔の幅を画素数で計算します。
        let faceWidth = face.boundingBox.width * CGFloat(image.width)
        // 元画像上の顔の高さを画素数で計算します。
        let faceHeight = face.boundingBox.height * CGFloat(image.height)
        // 64 ピクセル未満の顔は照合に使わず、呼び出し元で検出記録を残します。
        guard min(faceWidth, faceHeight) >= 64 else { return nil }
        // Vision が返した顔パーツの位置を取り出します。
        guard let landmarks = face.landmarks,
              // 左目の位置が存在することを確認します。
              let leftEye = landmarks.leftEye,
              // 右目の位置が存在することを確認します。
              let rightEye = landmarks.rightEye,
              // 鼻先の位置が存在することを確認します。
              let nose = landmarks.noseCrest ?? landmarks.nose,
              // 外側の唇の輪郭が存在することを確認します。
              let lips = landmarks.outerLips else { return nil }
        // 各目の輪郭中心を顔内の正規化座標にします。
        guard let left = center(leftEye), let right = center(rightEye), let tip = lastPoint(nose), let mouth = lipCorners(lips) else { return nil }
        // 上下関係が崩れたランドマークを照合に使わず未分類へ残します。
        guard left.x < right.x, (left.y + right.y) / 2 > tip.y, tip.y > (mouth.0.y + mouth.1.y) / 2 else { return nil }
        // 顔領域内の座標を、元画像のピクセル座標に変換します。
        let observed = [left, right, tip, mouth.0, mouth.1].map { point in CGPoint(x: (face.boundingBox.minX + point.x * face.boundingBox.width) * CGFloat(image.width), y: (face.boundingBox.minY + point.y * face.boundingBox.height) * CGFloat(image.height)) }
        // Open Model Zoo が公開する五点の整列先を 128 ピクセルに変換します。
        let reference = [CGPoint(x: 0.31556875, y: 0.4615741071), CGPoint(x: 0.6826229167, y: 0.4615741071), CGPoint(x: 0.5002625, y: 0.6405053571), CGPoint(x: 0.349471875, y: 0.8246919643), CGPoint(x: 0.6534364583, y: 0.8246919643)].map { CGPoint(x: $0.x * 128, y: (1 - $0.y) * 128) }
        // 元画像の下向き座標系と参照画像の上向き座標系を合わせて変換します。
        guard let transform = similarity(from: observed, to: reference) else { return nil }
        // 位置合わせ後の五点と基準点のずれを二乗平均平方根で調べます。
        let residual = sqrt(zip(observed, reference).reduce(CGFloat.zero) { sum, pair in let point = pair.0.applying(transform); return sum + pow(point.x - pair.1.x, 2) + pow(point.y - pair.1.y, 2) } / 5)
        // 大きく崩れた整列結果から人物を推測しません。
        guard residual.isFinite, residual <= 16 else { return nil }
        // RGBA の 128×128 画素をゼロで初期化します。
        var pixels = [UInt8](repeating: 0, count: 128 * 128 * 4)
        // 配列のメモリを描画先として使います。
        let rendered = pixels.withUnsafeMutableBytes { raw -> Bool in
            // CoreGraphics の RGBA 描画コンテキストを作ります。
            guard let context = CGContext(data: raw.baseAddress, width: 128, height: 128, bitsPerComponent: 8, bytesPerRow: 128 * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            // 白を背景にし、顔の周辺が透明にならないようにします。
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            // 128 ピクセル四方の背景を塗ります。
            context.fill(CGRect(x: 0, y: 0, width: 128, height: 128))
            // 元画像から整列画像への相似変換を適用します。
            context.concatenate(transform)
            // 向き補正済みの動画フレームを描画します。
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            // 描画が成功したことを示します。
            return true
        // ここで直前の処理の範囲を閉じます。
        }
        // 描画できなければこの顔は処理しません。
        guard rendered else { return nil }
        // 3 色の各平面を 128×128 個の Float で用意します。
        var bgr = [Float](repeating: 0, count: 3 * 128 * 128)
        // 各画素を順に BGR の平面順へ並べ替えます。
        for index in 0..<(128 * 128) {
            // 青を最初の平面へ格納します。
            bgr[index] = Float(pixels[index * 4 + 2])
            // 緑を二番目の平面へ格納します。
            bgr[128 * 128 + index] = Float(pixels[index * 4 + 1])
            // 赤を三番目の平面へ格納します。
            bgr[2 * 128 * 128 + index] = Float(pixels[index * 4])
        // ここで直前の処理の範囲を閉じます。
        }
        // 整列後の画素からサムネイル用の画像を作ります。
        let provider = CGDataProvider(data: Data(pixels) as CFData)
        // CoreGraphics 画像に変換します。
        guard let provider, let crop = CGImage(width: 128, height: 128, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 128 * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent), let data = jpeg(crop, quality: 0.8) else { return nil }
        // 推論入力と画面用画像を一緒に返します。
        return (bgr, data)
    // ここで直前の処理の範囲を閉じます。
    }

    // ランドマーク輪郭の平均位置を返します。
    private static func center(_ region: VNFaceLandmarkRegion2D) -> CGPoint? {
        // 点がなければ位置を決められません。
        guard region.pointCount > 0 else { return nil }
        // 全点の横座標を合計します。
        let x = region.normalizedPoints.reduce(CGFloat.zero) { $0 + $1.x }
        // 全点の縦座標を合計します。
        let y = region.normalizedPoints.reduce(CGFloat.zero) { $0 + $1.y }
        // 点数で割って中心を返します。
        return CGPoint(x: x / CGFloat(region.pointCount), y: y / CGFloat(region.pointCount))
    // ここで直前の処理の範囲を閉じます。
    }

    // 鼻の輪郭の最下点を鼻先の近似として返します。
    private static func lastPoint(_ region: VNFaceLandmarkRegion2D) -> CGPoint? {
        // Vision の座標では鼻先が輪郭の低い位置にあります。
        region.normalizedPoints.min { $0.y < $1.y }
    // ここで直前の処理の範囲を閉じます。
    }

    // 唇の輪郭の左右端を口角として返します。
    private static func lipCorners(_ region: VNFaceLandmarkRegion2D) -> (CGPoint, CGPoint)? {
        // 輪郭が空なら口角を求められません。
        guard let left = region.normalizedPoints.min(by: { $0.x < $1.x }), let right = region.normalizedPoints.max(by: { $0.x < $1.x }) else { return nil }
        // 左右の端点を返します。
        return (left, right)
    // ここで直前の処理の範囲を閉じます。
    }

    // 五点の対応から拡大・回転・平行移動を最小二乗で求めます。
    private static func similarity(from source: [CGPoint], to target: [CGPoint]) -> CGAffineTransform? {
        // 両配列に同数の対応点があることを確かめます。
        guard source.count == target.count, source.count >= 2 else { return nil }
        // 元の五点の横方向平均を求めます。
        let sx = source.reduce(CGFloat.zero) { $0 + $1.x } / CGFloat(source.count)
        // 元の五点の縦方向平均を求めます。
        let sy = source.reduce(CGFloat.zero) { $0 + $1.y } / CGFloat(source.count)
        // 目標の五点の横方向平均を求めます。
        let tx = target.reduce(CGFloat.zero) { $0 + $1.x } / CGFloat(target.count)
        // 目標の五点の縦方向平均を求めます。
        let ty = target.reduce(CGFloat.zero) { $0 + $1.y } / CGFloat(target.count)
        // 回転と拡大に使う分母を初期化します。
        var denominator = CGFloat.zero
        // 横向きの相関値を初期化します。
        var real = CGFloat.zero
        // 縦向きの相関値を初期化します。
        var imaginary = CGFloat.zero
        // 五点を一つずつ集計します。
        for index in source.indices {
            // 元の点を中心からの差に変換します。
            let x = source[index].x - sx
            // 元の縦位置を中心からの差に変換します。
            let y = source[index].y - sy
            // 目標の横位置を中心からの差に変換します。
            let u = target[index].x - tx
            // 目標の縦位置を中心からの差に変換します。
            let v = target[index].y - ty
            // 元の点の二乗距離を足します。
            denominator += x * x + y * y
            // 拡大・回転の実数成分を足します。
            real += x * u + y * v
            // 拡大・回転の虚数成分を足します。
            imaginary += x * v - y * u
        // ここで直前の処理の範囲を閉じます。
        }
        // 点が重なっている場合は変換できません。
        guard denominator > 0 else { return nil }
        // 回転と拡大の実数係数を求めます。
        let a = real / denominator
        // 回転と拡大の虚数係数を求めます。
        let b = imaginary / denominator
        // 元の中心を目標の中心へ移す横移動量を求めます。
        let moveX = tx - a * sx + b * sy
        // 元の中心を目標の中心へ移す縦移動量を求めます。
        let moveY = ty - b * sx - a * sy
        // CoreGraphics の相似変換として返します。
        return CGAffineTransform(a: a, b: b, c: -b, d: a, tx: moveX, ty: moveY)
    // ここで直前の処理の範囲を閉じます。
    }

    // 向き補正済みフレームから顔の矩形部分を安全に切り出します。
    private static func faceCrop(_ image: CGImage, face: VNFaceObservation) -> Data? {
        // Vision の正規化領域を取り出します。
        let box = face.boundingBox
        // 不正な座標が画像切り出しへ渡らないようにします。
        guard box.minX.isFinite, box.minY.isFinite, box.width.isFinite, box.height.isFinite else { return nil }
        // 画像のピクセル領域を定義します。
        let imageBounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        // Vision の左下原点を CGImage の左上原点へ変換します。
        let requested = CGRect(x: box.minX * CGFloat(image.width), y: (1 - box.maxY) * CGFloat(image.height), width: box.width * CGFloat(image.width), height: box.height * CGFloat(image.height))
        // 画像外の座標を切り落としてピクセル境界へ丸めます。
        let clipped = requested.standardized.intersection(imageBounds).integral.intersection(imageBounds)
        // 画素がない領域は切り出せません。
        guard !clipped.isNull, clipped.width >= 1, clipped.height >= 1 else { return nil }
        // 指定した領域の画像を作ります。
        guard let croppedImage = image.cropping(to: clipped) else { return nil }
        // 画面表示と保存に使う JPEG を返します。
        return jpeg(croppedImage, quality: 0.8, maximumSide: 192)
    // ここで直前の処理の範囲を閉じます。
    }

    // CGImage を指定品質の JPEG データに変換します。
    private static func jpeg(_ image: CGImage, quality: CGFloat, maximumSide: CGFloat = 1280) -> Data? {
        // 保存用画像を必要な大きさまで縮めて容量と表示負荷を減らします。
        let scale = min(1, maximumSide / CGFloat(max(image.width, image.height)))
        // 実際に保存する画像を保持します。
        var output = image
        // 縮小が必要な画像だけ描画し直します。
        if scale < 1 {
            // 縦横比を維持した描画先を確保します。
            guard let context = CGContext(data: nil, width: max(1, Int(CGFloat(image.width) * scale)), height: max(1, Int(CGFloat(image.height) * scale)), bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
            // 元画像を縮小して描画します。
            context.draw(image, in: CGRect(x: 0, y: 0, width: context.width, height: context.height))
            // 保存対象を縮小後の画像へ切り替えます。
            guard let scaled = context.makeImage() else { return nil }
            // 縮小結果を保持します。
            output = scaled
        // 縮小処理を閉じます。
        }
        // AppKit が扱えるビットマップ画像を作ります。
        let bitmap = NSBitmapImageRep(cgImage: output)
        // JPEG の圧縮率を指定してバイナリを返します。
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: quality])
    // ここで直前の処理の範囲を閉じます。
    }
// ここで直前の処理の範囲を閉じます。
}
