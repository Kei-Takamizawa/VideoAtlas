#!/bin/zsh
# このファイルのある Scripts フォルダへ移動します。
cd "${0:A:h}"
# リポジトリのルートへ移動します。
cd ..
# Python版専用の仮想環境があるか確認します。
if [[ ! -x .venv/bin/python ]]; then
  # 未セットアップのときは必要な操作を表示します。
  print '先に README.md の Python 版セットアップを実行してください。'
  # 明示的な失敗番号で終了します。
  exit 1
# 仮想環境の確認条件を閉じます。
fi
# このリポジトリの Python 版 VideoAtlas を起動します。
.venv/bin/python -m videoatlas
