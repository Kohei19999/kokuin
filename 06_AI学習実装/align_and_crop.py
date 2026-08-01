import cv2
import numpy as np
import math
import glob
import os
from ultralytics import YOLO

def find_coin_center(image):
    # グレースケール変換と平滑化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # ハフ変換で円を検出（コインの外形）
    # パラメータは50円玉のサイズに合わせて調整
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=100, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 最初の円（一番大きな円）の中心を返す
        cx, cy, r = circles[0][0]
        return cx, cy, r
    return None, None, None

def align_coin(img_path, model, output_dir):
    print(f"🔄 処理中: {os.path.basename(img_path)}")
    img = cv2.imread(img_path)
    if img is None:
        return
        
    # 1. コインの中心を探す（OpenCV）
    cx, cy, r = find_coin_center(img)
    if cx is None:
        print("  ⚠️ コインの輪郭（中心）が検出できませんでした。")
        return

    # 2. YOLOで年号の位置を探す（AI）
    results = model(img, verbose=False)
    if len(results[0].boxes) == 0:
        print("  ⚠️ 年号が検出できませんでした。")
        return
        
    # 最も信頼度の高い検出結果を取得
    box = results[0].boxes[0]
    # バウンディングボックスの中心を計算
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    yx = (x1 + x2) / 2.0
    yy = (y1 + y2) / 2.0
    
    # 3. 回転角度の計算（アークタンジェント）
    # コイン中心(cx, cy)から年号中心(yx, yy)への角度を計算
    # OpenCVのY軸は下向きが正
    theta_rad = math.atan2(yy - cy, yx - cx)
    theta_deg = math.degrees(theta_rad)
    
    # 年号が常に「真下（90度）」に来るように回転角を計算
    # cv2.warpAffineの回転は反時計回りが正なので、(現在の角度 - 90度)だけ回す
    rotation_angle = theta_deg - 90
    
    # 4. 画像の回転
    h, w = img.shape[:2]
    # 中心(cx, cy)を基準に回転
    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    
    # --- デバッグ用の描画 ---
    debug_img = rotated_img.copy()
    # 中心点と、真下（年号があるべき場所）に線を引く
    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(debug_img, (cx, cy), (cx, cy + r), (0, 255, 0), 2)
    
    # 5. 年号部分の切り出し（クロップ）
    # 真下にある年号の領域を固定座標で切り出す（半径rを元に調整）
    # ※パラメーターは要調整
    crop_w = int(r * 1.2)
    crop_h = int(r * 0.5)
    crop_x1 = int(cx - crop_w / 2)
    crop_y1 = int(cy + r * 0.4)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    # 画像の範囲外に出ないようにクリップ
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)
    
    cropped_img = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 6. 保存
    base_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, f"aligned_{base_name}"), debug_img)
    if cropped_img.size > 0:
        cv2.imwrite(os.path.join(output_dir, f"cropped_{base_name}"), cropped_img)
    print("  ✅ アライメントと切り出し完了！")

if __name__ == "__main__":
    # 出力先フォルダの作成
    output_dir = "runs/align_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルの読み込み
    model_path = r"runs\detect\50yen_run\weights\best.pt"
    print("🤖 YOLOモデルを読み込み中...")
    model = YOLO(model_path)
    
    # 対象画像の読み込み（テスト用のデータセットから）
    test_dir = r"dataset\test\images"
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    for img_path in image_paths:
        align_coin(img_path, model, output_dir)
        
    print(f"\n🎉 すべての処理が完了しました！ 結果は {output_dir} を確認してください。")
