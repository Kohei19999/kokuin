import cv2
import numpy as np
import math
import glob
import os

def find_coin_center(gray):
    # ハフ変換で円を検出
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=100, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        return cx, cy, r
    return None, None, None

def imread_japanese(path):
    with open(path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def imwrite_japanese(path, img):
    ext = os.path.splitext(path)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(path, mode='w+b') as f:
            n.tofile(f)

def align_coin_v2(img_path, template_gray, output_dir):
    print(f"🔄 処理中: {os.path.basename(img_path)}")
    img = imread_japanese(img_path)
    if img is None:
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # 1. コインの中心を探す（OpenCV）
    cx, cy, r = find_coin_center(gray)
    if cx is None:
        print("  ⚠️ コインの輪郭（中心）が検出できませんでした。")
        return

    # 2. パターンマッチングで「50」の位置を探す（360度スキャン）
    # 対象のコインがどんな角度か不明なので、画像を360度回転させながら
    # 最も「50」のテンプレートと一致する角度を探す（産業用カメラの定番手法）
    best_max_val = -1
    best_rotation = 0
    best_loc = (0, 0)
    
    h, w = gray.shape
    for angle in range(0, 360, 5): # 5度刻みで360度スキャン
        M_scan = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_gray = cv2.warpAffine(gray, M_scan, (w, h))
        
        res = cv2.matchTemplate(rotated_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_max_val:
            best_max_val = max_val
            best_rotation = angle
            best_loc = max_loc

    if best_max_val < 0.3:
        print(f"  ⚠️ '50'が見つかりませんでした (最高信頼度: {best_max_val:.2f})")
        return
        
    print(f"  🔍 '50'を発見！ 補正角度: {best_rotation}度 (信頼度: {best_max_val:.2f})")
    
    # 3. 最適な角度で画像を回転（これで「50」が必ず上に来る）
    M_final = cv2.getRotationMatrix2D((cx, cy), best_rotation, 1.0)
    rotated_img = cv2.warpAffine(img, M_final, (w, h))
    
    # --- デバッグ用の描画 ---
    debug_img = rotated_img.copy()
    cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)
    # 「50」があるはずの真上と、年号があるはずの真下に線を引く
    cv2.line(debug_img, (cx, cy), (cx, cy - r), (255, 0, 0), 2) # 青線（上：50）
    cv2.line(debug_img, (cx, cy), (cx, cy + r), (0, 255, 0), 2) # 緑線（下：年号）
    
    # 5. 年号部分の切り出し（クロップ）
    # 50が真上に来たため、年号は必ず「真下」にある。
    crop_w = int(r * 1.2)
    crop_h = int(r * 0.5)
    crop_x1 = int(cx - crop_w / 2)
    crop_y1 = int(cy + r * 0.4)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    # クリップ
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)
    
    cropped_img = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 6. 保存
    base_name = os.path.basename(img_path)
    imwrite_japanese(os.path.join(output_dir, f"aligned_v2_{base_name}"), debug_img)
    if cropped_img.size > 0:
        imwrite_japanese(os.path.join(output_dir, f"cropped_v2_{base_name}"), cropped_img)
    print("  ✅ アライメントと切り出し完了！")

if __name__ == "__main__":
    output_dir = "runs/align_results_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # テンプレートの読み込み
    template_path = "template_50.jpg"
    template_img = cv2.imread(template_path)
    if template_img is None:
        print("❌ template_50.jpg が見つかりません。先に setup_template.py を実行してください。")
        exit()
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # テスト画像で実行
    test_dir = r"dataset\test\images"
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    for img_path in image_paths:
        align_coin_v2(img_path, template_gray, output_dir)
        
    print(f"\n🎉 処理完了！ 結果: {output_dir}")
