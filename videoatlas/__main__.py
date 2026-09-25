# 画面を起動する関数を読み込みます。
from .ui import main

# このパッケージが直接実行された場合だけ画面を開きます。
if __name__ == "__main__":
    # Qtアプリの終了番号をOSへ返します。
    raise SystemExit(main())
