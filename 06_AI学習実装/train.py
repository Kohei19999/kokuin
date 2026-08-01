from ultralytics import YOLO
import os

if __name__ == "__main__":
    print("🚀 YOLOv8の学習を開始します...")

    # データセットのパス（絶対パスで指定）
    DATA_YAML = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\dataset\data.yaml"

    # 事前学習済みの軽量モデル（yolov8n = nano。最速で小さいモデル）をベースに学習
    model = YOLO("yolov8n.pt")

    # 学習の実行
    results = model.train(
        data=DATA_YAML,   # データセット設定ファイル
        epochs=50,        # 50回繰り返し学習
        imgsz=640,        # 入力画像サイズ
        batch=4,          # 画像枚数が少ないので小さめに設定
        name="50yen_run", # 結果の保存フォルダ名
        patience=20,      # 20エポック改善なければ早期終了
    )

    print("\n✅ 学習完了！")
    print(f"学習済みモデルの保存場所: runs/detect/50yen_run/weights/best.pt")
