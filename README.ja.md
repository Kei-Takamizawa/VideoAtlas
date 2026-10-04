# VideoAtlas

[English](README.md) | **日本語** | [简体中文](README.zh.md) | [हिन्दी](README.hi.md) | [Español](README.es.md) | [العربية](README.ar.md) | [Français](README.fr.md) | [Bahasa Indonesia](README.id.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Português](README.pt.md)

VideoAtlas は、Windows 上で動画を整理し、登場人物から見たい場面を探せるローカルアプリです。MP4、MOV、AVI、MKV、M4V、WebM に対応します。動画の内容ハッシュを使うため、名前を変更しただけでは再解析しません。

## できること

- 動画ごとに複数の顔サンプルを集め、動画内の人物を追跡
- 別々の動画に登場する人物の候補を慎重に照合
- 確認キューで同一人物・別人・後で確認を選択（S・D・L キー）
- 人物名、統合、分割、除外、代表画像を手動で編集
- 確認済みペアを記録し、照合結果を評価

動画間の自動統合は初期設定でオフです。提供された少数の実動画で照合を評価しましたが、現在の閾値では同一人物の取りこぼしが多く、実際のマスク着用時の精度は未確認です。ファイル名で人物を判定しません。検証結果と上半分モードの制限は[実装状況](docs/IMPLEMENTATION_STATUS.md)をご覧ください。

## Windows で始める

64 ビット Windows、Python 3.12、PowerShell が必要です。プロジェクトフォルダーで実行します。

```powershell
.\Scripts\setup_windows.cmd
.\Scripts\run_windows.cmd
```

標準設定は CPU を使います。セットアップではチェックサムを検証した公式モデルを取得します。英語と日本語の UI を選べます。[Windows セットアップ手順](docs/WINDOWS_SETUP.md)をご覧ください。

動画は移動されず、索引は PC に保存されます。動画や顔データはアップロードしません。モデルとソフトウェアには独自の利用条件があります。
