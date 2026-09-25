#!/usr/bin/env bash
# エラーや未設定の変数があれば処理を止め、壊れたモデルを配置しないようにします。
set -euo pipefail

# このスクリプトの一つ上のディレクトリをプロジェクトの場所として求めます。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 顔ランドマークモデルをアプリが読み込む場所を指定します。
MODEL_PATH="$PROJECT_DIR/videoatlas/resources/face_landmarker.task"
# Google の公式サンプルが使用する固定バージョンの配布先を指定します。
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
# 指定したモデル版の配布ファイルに対応する SHA-256 を指定します。
EXPECTED_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# 既にモデルがある場合は、正しいファイルか確認してから再取得を省きます。
if [[ -f "$MODEL_PATH" ]]; then
    # 期待するハッシュと一致すれば既存のモデルをそのまま使用します。
    if printf '%s  %s\n' "$EXPECTED_SHA256" "$MODEL_PATH" | shasum -a 256 --check --status; then
        # 利用可能なモデルの場所を表示します。
        printf '確認済み: %s\n' "$MODEL_PATH"
        # 正しいモデルがあるので処理を終了します。
        exit 0
    # 既存モデルの確認条件を閉じます。
    fi
    # 別のファイルを誤って上書きしないよう理由を表示します。
    printf '既存モデルの SHA-256 が一致しません: %s\n' "$MODEL_PATH" >&2
    # 手動でファイルを確認できるよう失敗として終了します。
    exit 1
# 既存モデルの有無を調べる条件を閉じます。
fi

# モデルを保存するディレクトリを用意します。
mkdir -p "$(dirname "$MODEL_PATH")"
# 検証前のファイルを最終名にしないため、一時ファイルを作ります。
TEMP_PATH="$(mktemp "$(dirname "$MODEL_PATH")/.face_landmarker.XXXXXX")"
# 途中で失敗したときも一時ファイルを残さないようにします。
trap 'rm -f "$TEMP_PATH"' EXIT
# 固定した公式 URL から一時ファイルへモデルを取得します。
curl --fail --location --silent --show-error --output "$TEMP_PATH" "$MODEL_URL"
# ダウンロードした内容の SHA-256 が既知の値に一致するか確認します。
printf '%s  %s\n' "$EXPECTED_SHA256" "$TEMP_PATH" | shasum -a 256 --check
# 検証済みモデルだけをアプリが読み込む名前へ移します。
mv "$TEMP_PATH" "$MODEL_PATH"
# 利用可能なモデルの場所を表示します。
printf '取得完了: %s\n' "$MODEL_PATH"
