import cv2
import numpy as np
import os
import pandas as pd
import math
import shutil

# ── ゼロベースから「過学習しないバランスの取れたベストプラクティス (try_021ベース)」を全30枚対応で構築 ──
# - コイン外周は MedianBlur + HoughCircles (動的サイズ)
# - 穴有無判定は 絶対閾値(THRESH_BINARY_INV < 60) + ブラックハットの面積・円形度 (物理法則ベース)
# - 穴位置は マスクからの findContours 重心

def cv2_imread_jp(file_path):
    nparr = np.fromfile(file_path, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

def cv2_imwrite_jp(file_path, img):
    ext = os.path.splitext(file_path)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(file_path, mode='w+b') as f:
            n.tofile(f)

def find_coin_center(img, base_filename, preproc_out_dir):
    h, w = img.shape
    min_dim = min(h, w)
    coin_cx, coin_cy, coin_radius = w//2, h//2, min_dim//4
    
    # CLAHEは絶対にしない。ノイズを拾うため。
    blurred = cv2.medianBlur(img, 9)
    
    # 動的サイズでハフ変換
    min_r = int(min_dim * 0.15)
    max_r = int(min_dim * 0.45)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim//2,
        param1=100, param2=50, minRadius=min_r, maxRadius=max_r
    )
    
    if circles is not None and len(circles) > 0:
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            dist_to_center = math.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            if dist_to_center < min_dim * 0.3:
                coin_cx, coin_cy, coin_radius = cx, cy, r
                break
                
    # 途中経過保存: 外周検出結果
    color_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_debug, (coin_cx, coin_cy), coin_radius, (0, 255, 0), 2)
    cv2.drawMarker(color_debug, (coin_cx, coin_cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_01_outer.jpg"), color_debug)

    return coin_cx, coin_cy, coin_radius


def pipeline_A_check_hole_presence(img, coin_cx, coin_cy, coin_radius, base_filename, preproc_out_dir):
    h, w = img.shape
    roi_size = int(coin_radius * 1.5)
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(w, coin_cx + roi_size//2)
    y2 = min(h, coin_cy + roi_size//2)
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0, 0, 0
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_02_roi.jpg"), roi)
    
    # Step 1: 絶対的な暗さ（影）
    _, thresh_abs = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    
    mask_abs = np.zeros_like(thresh_abs)
    rh, rw = thresh_abs.shape
    cv2.circle(mask_abs, (rw//2, rh//2), int(coin_radius * 0.4), 255, -1)
    thresh_abs = cv2.bitwise_and(thresh_abs, mask_abs)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_03_thresh_abs.jpg"), thresh_abs)
    
    dark_abs = cv2.countNonZero(thresh_abs)
    # 面積の閾値も動的にする (半径の2乗に比例)
    abs_thresh_area = int((coin_radius**2) * 0.005)
    
    if dark_abs > abs_thresh_area:
        return 1, dark_abs, 0
        
    # Step 2: ブラックハット（局所的暗部）+ 円形度
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    thresh_bh = cv2.bitwise_and(thresh_bh, mask_abs)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_04_thresh_bh.jpg"), thresh_bh)
    
    contours, _ = cv2.findContours(thresh_bh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_hole = 0
    best_circ = 0.0
    
    min_area = int((coin_radius**2) * 0.002)
    
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area > min_area and perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.4:  # 丸ければ穴
                is_hole = 1
                best_circ = circularity
                
    return is_hole, dark_abs, int(best_circ * 100)


def pipeline_B_check_misalignment(img, coin_cx, coin_cy, coin_radius, base_filename, preproc_out_dir):
    h, w = img.shape
    roi_size = int(coin_radius * 0.6)  # 狭めに取って内枠の黒線を拾わないように
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(w, coin_cx + roi_size//2)
    y2 = min(h, coin_cy + roi_size//2)
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0, 0, 0, 0, 0
        
    _, thresh_abs = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    
    mask = cv2.bitwise_or(thresh_abs, thresh_bh)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hx, hy = -1, -1
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > (coin_radius**2)*0.002:
            max_area = area
            M = cv2.moments(c)
            if M["m00"] != 0:
                hx = int(M["m10"] / M["m00"]) + x1
                hy = int(M["m01"] / M["m00"]) + y1
                
    if hx == -1:
        return 0, 0, 0, 0, 0
        
    dist = math.sqrt((coin_cx - hx)**2 + (coin_cy - hy)**2)
    ratio = dist / coin_radius
    
    is_misaligned = 1 if ratio > 0.08 else 0
    
    # 途中経過保存: 最終判定結果
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, (coin_cx, coin_cy), coin_radius, (0, 255, 0), 2)
    cv2.circle(color_img, (hx, hy), int(coin_radius*0.1), (0, 0, 255), -1)
    cv2.drawMarker(color_img, (hx, hy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_05_final.jpg"), color_img)
    
    return is_misaligned, dist, ratio, hx, hy


def main():
    IMG_DIR = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    LABEL_CSV = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル_v2.csv"
    
    # 出力先を現在のディレクトリ（実験ベスト盤）の「実行結果」に固定
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_BASE = os.path.join(SCRIPT_DIR, "実行結果")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # 古い結果があれば一旦クリア
    for f in os.listdir(OUTPUT_BASE):
        f_path = os.path.join(OUTPUT_BASE, f)
        if os.path.isfile(f_path):
            os.remove(f_path)
    
    TRY_DIR = OUTPUT_BASE

    df_labels = pd.read_csv(LABEL_CSV, encoding='utf-8-sig')

    results = []
    
    print(f"--- 50円玉 穴＆ズレ検出 [実験ベスト盤: バランス型ベストプラクティス] ---")
    
    # 集計用
    correct_A = 0
    correct_B = 0
    correct_Total = 0
    total_count = 0

    for idx, row in df_labels.iterrows():
        filename = row['ファイル名']
        img_path = os.path.join(IMG_DIR, filename)
        
        gt_hole = row['中心穴の有無(1:あり/0:なし)']
        gt_misalign = row['穴のズレ(1:ズレあり/0:正常)']
        
        img = cv2_imread_jp(img_path)
        if img is None:
            continue
            
        total_count += 1
            
        coin_cx, coin_cy, coin_radius = find_coin_center(img, filename, TRY_DIR)
        
        # Pipeline A
        pred_hole, score1, score2 = pipeline_A_check_hole_presence(img, coin_cx, coin_cy, coin_radius, filename, TRY_DIR)
        
        # Pipeline B
        if pred_hole == 1:
            pred_misalign, dist, ratio, hx, hy = pipeline_B_check_misalignment(img, coin_cx, coin_cy, coin_radius, filename, TRY_DIR)
        else:
            pred_misalign, dist, ratio, hx, hy = 0, 0.0, 0.0, -1, -1

        results.append({
            "ファイル名": filename,
            "正解_穴有無": gt_hole,
            "予測_穴有無": pred_hole,
            "正解_穴ズレ": gt_misalign,
            "予測_穴ズレ": pred_misalign,
            "距離px": dist,
            "ズレ率": ratio,
            "coin_r": coin_radius
        })

        is_a_ok = (pred_hole == gt_hole) or (gt_hole == 0.5)
        is_b_ok = (pred_misalign == gt_misalign) or (gt_misalign == 0.5)
        
        if is_a_ok: correct_A += 1
        if pred_hole == 1 and not pd.isna(gt_misalign):
            if is_b_ok: correct_B += 1
            
        if is_a_ok and (pd.isna(gt_misalign) or is_b_ok):
            correct_Total += 1
            icon = "✅"
        else:
            icon = "❌"
            
        print(f"[{icon}] {filename:<20} | 穴: {pred_hole}/{gt_hole} | ズレ: {pred_misalign}/{gt_misalign} (dist:{dist:.1f}px ratio:{ratio:.3f} r:{coin_radius})")

    df_out = pd.DataFrame(results)
    csv_path = os.path.join(TRY_DIR, "evaluation_results.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    acc_A     = correct_A / total_count * 100 if total_count > 0 else 0
    # Pipeline B: count only samples where hole was predicted present
    total_B   = sum(1 for r in results if r['予測_穴有無'] == 1)
    acc_B     = correct_B / total_B * 100 if total_B > 0 else 0
    acc_Total = correct_Total / total_count * 100 if total_count > 0 else 0

    print(f"\n{'='*55}")
    print(f"  実験ベスト盤 [バランス型ベストプラクティス] 最終結果")
    print(f"  サンプル数: {total_count}枚")
    print(f"  パイプラインA 「穴の有無」精度: {acc_A:.1f}%  ({correct_A}/{total_count})正解")
    print(f"  パイプラインB 「穴のズレ」精度: {acc_B:.1f}%  ({correct_B}/{total_B})正解")
    print(f"  総合精度 (両方正解): {acc_Total:.1f}%  ({correct_Total}/{total_count})正解")
    print(f"{'='*55}")
    print(f"\n  途中経過画像 + 結果CSV: {TRY_DIR}")

if __name__ == "__main__":
    main()
