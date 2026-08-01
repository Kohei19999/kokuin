import cv2
import numpy as np
import os
import pandas as pd
import math
import shutil

def cv2_imread_jp(file_path):
    try:
        nparr = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def cv2_imwrite_jp(file_path, img):
    ext = os.path.splitext(file_path)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(file_path, mode='w+b') as f:
            n.tofile(f)

# =========================================================================
# パイプラインA: 【穴の有無判定】専用処理（外径検出に依存せず、絶対閾値で判定）
# =========================================================================
def pipeline_A_check_hole_presence(img, base_filename, preproc_out_dir):
    h, w = img.shape
    # 画像全体の中心から固定ROI（80x80）を切り出し
    roi_size = 80
    roi = img[h//2 - roi_size//2 : h//2 + roi_size//2, w//2 - roi_size//2 : w//2 + roi_size//2]
    
    # 元ROIの保存
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_pipelineA_1_roi.jpg"), roi)
    
    # 貫通穴は光が抜けたり奥が黒い「絶対的な暗さ」を持つため、固定の絶対閾値（50以下）で二値化
    # Otsuのような相対二値化を使わないことで、「穴なし」コインでの誤検知を防ぐ
    _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 中央マスク（四隅のノイズカット）
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (roi_size//2, roi_size//2), 30, 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_pipelineA_2_thresh.jpg"), thresh)
    
    # 暗い部分のピクセル数をカウント
    dark_pixels = cv2.countNonZero(thresh)
    
    # 250ピクセル以上なら「穴あり」
    is_hole = 1 if dark_pixels > 250 else 0
    return is_hole, dark_pixels


# =========================================================================
# パイプラインB: 【穴のズレ判定】専用処理（円形度フィルタ＋Otsu＋重心距離）
# =========================================================================
def pipeline_B_check_misalignment(img, base_filename, preproc_out_dir):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 1. 外径検出（CLAHE + メディアン + 円形度フィルタ）
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    blurred = cv2.medianBlur(enhanced, 9)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_pipelineB_1_edges.jpg"), edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coin_cx, coin_cy = img.shape[1]//2, img.shape[0]//2
    coin_radius = 0
    
    valid_coin_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area > 5000 and perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.5:
                valid_coin_contours.append((c, area))
                
    if valid_coin_contours:
        best_contour = max(valid_coin_contours, key=lambda x: x[1])[0]
        (x, y), radius = cv2.minEnclosingCircle(best_contour)
        coin_cx, coin_cy = int(x), int(y)
        coin_radius = int(radius)
        
    cv2.circle(color_img, (coin_cx, coin_cy), coin_radius, (255, 0, 0), 2)
    cv2.drawMarker(color_img, (coin_cx, coin_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 2. 穴の重心算出（Otsu）
    roi_size = 140
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(img.shape[1], coin_cx + roi_size//2)
    y2 = min(img.shape[0], coin_cy + roi_size//2)
    
    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
    _, thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (roi_size//2, roi_size//2), 45, 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)
    
    hole_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    is_misaligned = 0
    distance = 0.0
    
    valid_holes = []
    for hc in hole_contours:
        h_area = cv2.contourArea(hc)
        h_peri = cv2.arcLength(hc, True)
        if h_area > 150 and h_peri > 0:
            h_circ = 4 * math.pi * h_area / (h_peri * h_peri)
            if h_circ > 0.4:
                valid_holes.append((hc, h_area))
                
    if valid_holes:
        best_hole = max(valid_holes, key=lambda x: x[1])[0]
        M = cv2.moments(best_hole)
        if M["m00"] != 0:
            hole_cx_roi = int(M["m10"] / M["m00"])
            hole_cy_roi = int(M["m01"] / M["m00"])
            
            hole_cx = x1 + hole_cx_roi
            hole_cy = y1 + hole_cy_roi
            
            distance = math.sqrt((hole_cx - coin_cx)**2 + (hole_cy - coin_cy)**2)
            
            # ズレ判定（12ピクセル以上でズレあり）
            if distance > 12.0:
                is_misaligned = 1
                
            cv2.drawMarker(color_img, (hole_cx, hole_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.line(color_img, (coin_cx, coin_cy), (hole_cx, hole_cy), (0, 255, 255), 2)
            
    cv2.putText(color_img, f"Dist:{distance:.1f}px (Misalign:{is_misaligned})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_pipelineB_2_misalignment.jpg"), color_img)
        
    return is_misaligned, distance


def get_next_try_folder(base_dir):
    i = 1
    while True:
        folder_name = f"try_{i:03d}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1

if __name__ == "__main__":
    csv_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル.csv"
    img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    base_out_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\50円玉写真の前処理"
    
    os.makedirs(base_out_dir, exist_ok=True)
    trial_dir = get_next_try_folder(base_out_dir)
    print(f"--- 50円玉 実画像 穴＆ズレ検出 [{os.path.basename(trial_dir)}] (分離パイプライン構成) ---")
    
    current_script_path = os.path.abspath(__file__)
    shutil.copy(current_script_path, os.path.join(trial_dir, "evaluate_script.py"))
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    results = []
    
    for idx, row in df.iterrows():
        filename = row["ファイル名"]
        true_hole = row["中心穴の有無(1:あり/0:なし)"]
        true_misalign = row["穴のズレ(1:ズレあり/0:正常)"]
        
        img_path = os.path.join(img_dir, filename)
        img = cv2_imread_jp(img_path)
        if img is None: continue
        
        base_filename = os.path.splitext(filename)[0]
        
        # ---------------------------------------------------------
        # 独立した2つのパイプラインを個別に実行！
        # ---------------------------------------------------------
        # パイプラインA: 穴の有無判定
        pred_hole, dark_pixels = pipeline_A_check_hole_presence(img, base_filename, trial_dir)
        
        # パイプラインB: 穴のズレ判定（穴が存在する場合のみ計算）
        if pred_hole == 1:
            pred_misalign, dist = pipeline_B_check_misalignment(img, base_filename, trial_dir)
        else:
            pred_misalign, dist = 0, 0.0 # 穴がない場合はズレなし扱い
            
        ok_hole = (pred_hole == true_hole)
        ok_misalign = (pred_misalign == true_misalign) if true_hole == 1 else True
        
        mark = "✅" if (ok_hole and ok_misalign) else "❌"
        print(f"[{mark}] {filename} | 穴有無: 正解{true_hole}/予測{pred_hole} | ズレ: 正解{true_misalign}/予測{pred_misalign} (距離:{dist:.1f}px)")
        
        results.append({
            "ファイル名": filename,
            "正解_穴有無": true_hole,
            "予測_穴有無": pred_hole,
            "判定_穴有無": "OK" if ok_hole else "NG",
            "暗部面積(px)": dark_pixels,
            "正解_穴ズレ": true_misalign,
            "予測_穴ズレ": pred_misalign,
            "判定_穴ズレ": "OK" if ok_misalign else "NG",
            "ズレ距離(px)": round(dist, 1),
            "総合判定": "OK" if (ok_hole and ok_misalign) else "NG"
        })
        
    results_df = pd.DataFrame(results)
    csv_out_path = os.path.join(trial_dir, "evaluation_results.csv")
    results_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    
    hole_acc = (results_df["判定_穴有無"] == "OK").mean() * 100
    misalign_acc = (results_df["判定_穴ズレ"] == "OK").mean() * 100
    total_acc = (results_df["総合判定"] == "OK").mean() * 100
    
    print(f"\n✅ 処理完了！")
    print(f"- パイプラインA【穴の有無】精度: {hole_acc:.1f}%")
    print(f"- パイプラインB【穴のズレ】精度: {misalign_acc:.1f}%")
    print(f"- 総合判定 精度: {total_acc:.1f}%")
    print(f"\n結果を以下に保存しました:")
    print(f"-> {trial_dir}")
