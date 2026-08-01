import cv2
import numpy as np
import os
import random

# --- 1. ダミー画像の自動生成（実画像がないためのPoC用） ---
def generate_dummy_images(output_dir, num_images=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # ベースサイズと基本座標（設計図上の位置）
    W, H = 800, 600
    ANCHOR_X, ANCHOR_Y = 200, 300  # 位置ズレ補正用の目印（大きな穴やエッジ）の座標
    HOLE_COORDS = [
        (500, 200), # 穴1 (bit 3)
        (600, 200), # 穴2 (bit 2)
        (500, 300), # 穴3 (bit 1)
        (600, 300)  # 穴4 (bit 0)
    ]
    
    # テンプレート用画像の作成（ズレ補正の基準）
    template_img = np.ones((H, W), dtype=np.uint8) * 200
    cv2.circle(template_img, (ANCHOR_X, ANCHOR_Y), 50, 50, -1) # アンカー（黒い大穴）
    # テンプレート（アンカー部分）を切り出しておく
    template = template_img[ANCHOR_Y-60:ANCHOR_Y+60, ANCHOR_X-60:ANCHOR_X+60].copy()
    cv2.imwrite(os.path.join(output_dir, "template.jpg"), template)
    
    generated_data = []
    
    for i in range(num_images):
        # ランダムな位置ズレ（-30px 〜 +30px）
        dx = random.randint(-30, 30)
        dy = random.randint(-30, 30)
        
        # 0〜15のランダムな号機番号（4ビット）
        machine_num = random.randint(0, 15)
        # 4ビットの0/1リストに変換 (例: 13 -> [1, 1, 0, 1])
        bits = [(machine_num >> 3) & 1, (machine_num >> 2) & 1, (machine_num >> 1) & 1, machine_num & 1]
        
        # 画像描画（梨地のグレー背景）
        img = np.ones((H, W), dtype=np.uint8) * 180
        # ノイズ付加（ダイカスト表面のザラザラ感）
        noise = np.random.randint(0, 50, (H, W), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # ズレを加味してアンカーを描画
        cv2.circle(img, (ANCHOR_X + dx, ANCHOR_Y + dy), 50, 40, -1)
        
        # ズレを加味してドリル穴を描画
        for j, (hx, hy) in enumerate(HOLE_COORDS):
            if bits[j] == 1:
                # 穴あり（黒くくっきり）
                cv2.circle(img, (hx + dx, hy + dy), 15, 20, -1)
                
        file_path = os.path.join(output_dir, f"dummy_{i+1:02d}.jpg")
        cv2.imwrite(file_path, img)
        generated_data.append({"path": file_path, "actual_num": machine_num, "bits": bits})
        
    return template, HOLE_COORDS, ANCHOR_X, ANCHOR_Y, generated_data


# --- 2. 識別ドリル穴の検出・判定アルゴリズム（本番想定） ---
def detect_holes(image_path, template, base_hole_coords, base_anchor_x, base_anchor_y):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # ステップ1: テンプレートマッチングによる位置ズレ(dx, dy)の計算
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc がテンプレートの左上座標。中心座標に変換
    h, w = template.shape
    found_anchor_x = max_loc[0] + w // 2
    found_anchor_y = max_loc[1] + h // 2
    
    # 設計図上の基準座標との差分（ズレ量）を計算
    dx = found_anchor_x - base_anchor_x
    dy = found_anchor_y - base_anchor_y
    
    # デバッグ描画（アンカー）
    cv2.circle(color_img, (found_anchor_x, found_anchor_y), 50, (255, 0, 0), 2)
    
    # ステップ2 & 3: 4箇所のROI切り出しと穴判定
    detected_bits = []
    for i, (base_hx, base_hy) in enumerate(base_hole_coords):
        # ズレ量を足して、実際の穴の位置をピンポイント特定
        target_x = base_hx + dx
        target_y = base_hy + dy
        
        # 40x40ピクセルのROIを切り出し
        roi_size = 40
        x1, y1 = target_x - roi_size//2, target_y - roi_size//2
        x2, y2 = target_x + roi_size//2, target_y + roi_size//2
        roi = img[y1:y2, x1:x2]
        
        # 二値化（穴の黒い影を抽出）。暗い部分(50以下)を白(255)にする
        _, thresh = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
        
        # 黒い部分（二値化後の白ピクセル）の面積をカウント
        hole_pixels = cv2.countNonZero(thresh)
        
        # ピクセル数が一定以上（例: 200px以上）なら「穴あり」
        is_hole = 1 if hole_pixels > 200 else 0
        detected_bits.append(is_hole)
        
        # デバッグ描画
        color = (0, 255, 0) if is_hole else (0, 0, 255) # あり:緑, なし:赤
        cv2.rectangle(color_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(color_img, str(is_hole), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    # 4ビット配列から整数（号機番号）を計算
    machine_num = (detected_bits[0] << 3) | (detected_bits[1] << 2) | (detected_bits[2] << 1) | detected_bits[3]
    
    # 総合結果の描画
    cv2.putText(color_img, f"Machine: No.{machine_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(color_img, f"Offset: x={dx}, y={dy}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 結果保存
    out_dir = "runs/drill_holes"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result_" + os.path.basename(image_path))
    cv2.imwrite(out_path, color_img)
    
    return machine_num, detected_bits

if __name__ == "__main__":
    print("--- 識別ドリル穴判定 PoCスクリプト ---")
    data_dir = "runs/dummy_parts"
    template, base_coords, ax, ay, test_data = generate_dummy_images(data_dir, 5)
    print(f"✅ {len(test_data)}枚のダミー画像（位置ズレ＆ランダム穴）を生成しました。")
    
    print("\n--- OpenCVによる全自動判定開始 ---")
    success_count = 0
    for data in test_data:
        pred_num, pred_bits = detect_holes(data["path"], template, base_coords, ax, ay)
        
        is_correct = (pred_num == data["actual_num"])
        if is_correct: success_count += 1
        
        mark = "OK" if is_correct else "NG"
        print(f"[{mark}] {os.path.basename(data['path'])} | 正解: {data['actual_num']:02d} {data['bits']} -> AI予測: {pred_num:02d} {pred_bits}")
        
    print(f"\n✅ 判定終了！ 正答率: {success_count}/{len(test_data)} ({(success_count/len(test_data))*100:.1f}%)")
    print("出力結果は runs/drill_holes フォルダをご確認ください。")
