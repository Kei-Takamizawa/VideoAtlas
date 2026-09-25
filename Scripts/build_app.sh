#!/usr/bin/env bash
# 途中で失敗した場合に未完成のアプリを完成扱いにしないため終了します。
set -euo pipefail
# このスクリプトが置かれたディレクトリの一つ上をプロジェクトの場所にします。
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# 作業するディレクトリをプロジェクトへ移します。
cd "$PROJECT_DIR"
# Swift と Clang が書き込むモジュールキャッシュを作業フォルダ内へ置きます。
export CLANG_MODULE_CACHE_PATH="$PROJECT_DIR/.build/module-cache"
# Swift Package Manager のモジュールキャッシュも同じ場所へ置きます。
export SWIFTPM_MODULECACHE_OVERRIDE="$PROJECT_DIR/.build/module-cache"
# パッケージ用のキャッシュディレクトリを用意します。
mkdir -p "$PROJECT_DIR/.build/module-cache" "$PROJECT_DIR/.build/swiftpm-cache"
# 最適化したアプリ本体とリソースをビルドします。
swift build --disable-sandbox --configuration release -debug-info-format none --cache-path "$PROJECT_DIR/.build/swiftpm-cache"
# Swift Package Manager に生成物の保存先を問い合わせます。
BIN_DIR="$(swift build --disable-sandbox --configuration release -debug-info-format none --cache-path "$PROJECT_DIR/.build/swiftpm-cache" --show-bin-path)"
# 出力する macOS アプリの場所を指定します。
APP_DIR="$PROJECT_DIR/dist/VideoAtlas.app"
# 前回の生成物だけを削除して新しいアプリを用意します。
rm -rf "$APP_DIR"
# macOS が認識するアプリの内部ディレクトリを作成します。
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
# ビルドされた実行ファイルをアプリ内へコピーします。
ditto "$BIN_DIR/VideoAtlas" "$APP_DIR/Contents/MacOS/VideoAtlas"
# 顔モデルと OpenVINO を含むリソースバンドルをコピーします。
ditto "$BIN_DIR/VideoAtlas_VideoAtlas.bundle" "$APP_DIR/Contents/Resources/VideoAtlas_VideoAtlas.bundle"
# Finder と macOS が読むアプリ情報をコピーします。
ditto "$PROJECT_DIR/Packaging/Info.plist" "$APP_DIR/Contents/Info.plist"
# 設定ファイルの形式が正しいことを検証します。
plutil -lint "$APP_DIR/Contents/Info.plist"
# JIT 権限を付けてローカル実行用の署名を行います。
codesign --force --sign - --options runtime --entitlements "$PROJECT_DIR/Packaging/VideoAtlas.entitlements" "$APP_DIR"
# 生成したアプリの署名を検証します。
codesign --verify --verbose "$APP_DIR"
# 完成したアプリの絶対パスを表示します。
printf '完成: %s\n' "$APP_DIR"
