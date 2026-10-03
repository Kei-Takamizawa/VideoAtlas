# VideoAtlas

[English](README.md) | **日本語** | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

**Windows 専用のローカル動画ライブラリです。** 動画を閲覧・再生し、登場する人物を手がかりにシーンを探せます。顔の自動候補はアプリ内で確認・修正できます。

## できること

- 選択したフォルダー内の動画を一覧表示して再生
- 人物候補、関連動画、顔が登場する時刻を検索
- 解析を一時停止・再開し、顔の割り当て・分割・統合・除外を手動で修正
- 元動画を移動せず、ライブラリ情報を PC に保存

自動グループは見た目の類似度に基づく候補で、誤る場合があります。結果を確認・修正してください。顔照合の精度は測定していません。処理はローカルで行い、初回セットアップ時に必要なソフトウェアとモデルを取得します。

## Windows で始める

64 ビット Windows、Python 3.12、PowerShell が必要です。プロジェクトフォルダーで実行します。

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

標準設定は CPU で動作します。NVIDIA GPU の CUDA を使う場合は `.\Scripts\setup_windows.cmd -Gpu` を実行してください。TensorRT は別途 NVIDIA からインストールします。[詳しい手順](docs/WINDOWS_SETUP.md)。Windows 実装は実行時の検証をしていません。テスト、アプリ起動、実動画での解析は行っていません。

## プライバシーとモデル

索引と生成画像は Windows のローカルアプリデータに保存されます。動画や顔データはアップロードしません。セットアップ時に固定版の FACE01 日本人顔モデルと MediaPipe 顔ランドマークモデルを取得し、SHA-256 を確認します。FACE01 のモデルには別途利用条件があります。利用前に確認してください。プロジェクトの [MIT ライセンス](LICENSE)は第三者モデルや依存ソフトウェアには適用されません。

## 状態

Windows 実装は実行時の検証をしていません。テスト、アプリ起動、実動画での解析は行っていません。この Windows 専用版では以前の macOS アプリとセットアップはサポートしません。
