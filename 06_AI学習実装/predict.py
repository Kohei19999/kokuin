from ultralytics import YOLO
import glob
import os

if __name__ == "__main__":
    # 学習済みのベストモデルを読み込む
    MODEL_PATH = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\runs\detect\50yen_run\weights\best.pt"
    model = YOLO(MODEL_PATH)

    # テスト用画像（dataset/test フォルダの画像）
    TEST_DIR = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\dataset\test\images"
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpeg")) + glob.glob(os.path.join(TEST_DIR, "*.jpg"))

    if not test_images:
        print("⚠️ テスト画像が見つかりません。")
    else:
        print(f"🔍 {len(test_images)}枚の画像でテスト推論を実行します...")
        results = model(
            test_images,
            save=True,          # 検出枠を描画した画像を保存
            save_txt=True,      # 検出結果をtxtファイルにも保存
            conf=0.3,           # 信頼度30%以上の結果を表示
        )

        print("\n✅ 推論完了！結果は以下のフォルダに保存されました：")
        print(r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\runs\detect\predict")
        
        # 結果を表示
        for r in results:
            print(f"\n📷 {os.path.basename(r.path)}")
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                print(f"   → 検出: '{r.names[cls]}' (信頼度: {conf*100:.1f}%)")
