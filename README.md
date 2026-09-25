# VideoAtlas

日本語 | [English](README.en.md)

VideoAtlas は、macOS 向けのローカル動画ライブラリアプリです。指定したフォルダの動画を一覧・再生し、検出した顔を手がかりに人物候補、関連動画、登場時刻を探せます。現行アプリは Python と PySide6 で実装されています。

## できること

- 動画フォルダを追加し、MP4、MOV、M4V をサブフォルダまで一覧にする
- 動画を再生し、検出時刻から再生位置へ移動する
- 人物候補の顔画像から、関連する動画と検出時刻を絞り込む
- 顔を人物候補に割り当てる、人物候補を分割・統合する、顔を未分類へ戻す、または除外する
- 解析を一時停止・再開し、解析結果を SQLite に保存する

自動分類に迷いがある顔は未分類に残し、必要に応じて手動で修正できます。人物候補はモデルの類似度に基づくもので、本人確認や生体認証を行うものではありません。

| 日本語 UI | English UI |
| --- | --- |
| ![日本語の初期画面](Screenshots/python_initial_screen.png) | ![English initial screen](Screenshots/python_english_screen.png) |

## 動作環境と起動

セットアップ手順は Apple Silicon 搭載の Mac と Python 3.14 を対象にしています。依存パッケージをインストールした後、動画解析、顔処理、再生、保存はローカルで行います。他の OS、CPU 構成、Python バージョンでの動作は未検証です。

```bash
# Python 3.14 を使う仮想環境を作成します。
python3 -m venv .venv
# アプリの依存パッケージを仮想環境へインストールします。
.venv/bin/python -m pip install -r requirements-python.txt
# Google 公式配布元から顔ランドマークモデルを取得し、SHA-256を確認します。
Scripts/download_face_landmarker.sh
# VideoAtlas を起動します。
.venv/bin/python -m videoatlas
```

次回以降は `Scripts/run_python_app.command` から起動できます。

## Swift 参考実装

`Sources/` と `Package.swift` には Swift の参考実装があります。`Scripts/build_app.sh` はこの参考実装をローカルでビルドし、`dist/VideoAtlas.app` を生成します。現行の Python アプリの起動手順は上記のセットアップを参照してください。

## 使い方

1. 「フォルダを追加」を選ぶか、動画フォルダをアプリへドラッグします。
2. 動画の追加・変更を反映するには「フォルダを更新」を選びます。
3. 「人物」で顔写真を選ぶと、関連する動画と検出時刻が表示されます。時刻からその位置を再生できます。
4. 顔写真のコンテキストメニューで、人物候補への割り当て、分割、未分類への変更、除外を行えます。人物候補は名前変更や統合ができます。
5. 左下の言語メニューで日本語または英語を選べます。選択は保存され、次回起動後も使われます。初期設定は日本語です。

通常解析は 2 秒ごとにフレームを確認し、長辺を最大 1280 ピクセルに縮小します。詳細再解析は 0.5 秒ごとに元解像度のフレームを確認します。検出時刻一覧は一つの動画につき最初の 60 件を表示し、追加読み込みができます。顔の編集は解析を一時停止してから行います。

## データとプライバシー

- Python 版の索引と生成画像は `~/Library/Application Support/VideoAtlasPython/` に保存します。
- 元動画は元の場所に残し、アプリは移動しません。
- 「インデックスを消去」は Python 版の索引と生成画像を削除します。元動画は削除しません。
- Swift 版のデータは Python 版へ自動移行しません。
- アプリの解析処理に、顔データや動画を外部へ送信する機能はありません。初回セットアップでは依存パッケージと顔ランドマークモデルをインターネットから取得します。

OpenVINO は[匿名の利用状況テレメトリ](https://docs.openvino.ai/2026/about-openvino/additional-resources/telemetry.html)について案内しています。停止したい場合は、依存パッケージのインストール後に `.venv/bin/opt_in_out --opt_out` を実行できます。

## 精度と検証範囲

顔の検出や人物候補の自動分類は、動画の画質、顔の大きさ、向き、照明などで結果が変わります。正解ラベルを用いた人物識別の適合率・再現率や、マスク着用時の識別率は評価していません。別人が同じ候補に入ることや、同じ人物が別候補に分かれることがあります。重要な判断には自動分類だけを使用せず、手動で結果を確認してください。

Python アプリについては、構文検査、空のライブラリでの画面起動、日英の即時切替、言語設定の再読み込み、既知の解析状態文の翻訳を確認しています。モデル取得スクリプトは、空の一時ディレクトリでダウンロードと SHA-256 検証を確認しています。実動画を用いた Python 版の解析結果、人物識別精度、他の環境での動作は未検証です。Swift 参考実装の検証結果は Python アプリの精度を示しません。

## 使用技術とモデル

- Python、PySide6、OpenCV、MediaPipe、OpenVINO、SQLite
- 顔ランドマーク: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)。モデルファイルはセットアップ時に[Googleの公式配布元](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)から取得し、リポジトリには含めません。SHA-256 は `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` です。
- 顔特徴量: [Open Model Zoo face-reidentification-retail-0095](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-reidentification-retail-0095)
- Python 依存関係の版: [`requirements-python.txt`](requirements-python.txt)

このプロジェクトのソースコードは [MIT License](LICENSE) で公開します。この許諾は第三者のモデル、ランタイム、ライブラリには適用されません。Swift 版の OpenVINO 配布物に関するライセンス文書は `Sources/VideoAtlas/Resources/Licenses/` にあります。Python 版でダウンロードする Google のモデルファイルについては、ソースコードとは別の配布条件が適用されます。モデルファイルの正確なバージョンと SHA-256 を固定していますが、そのファイル固有の再配布条件は確認できていません。
