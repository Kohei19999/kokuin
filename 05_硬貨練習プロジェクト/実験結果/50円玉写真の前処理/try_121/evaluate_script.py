import cv2
import numpy as np
import os
import pandas as pd
import math
import shutil
import sys

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

def detect_center_hole_and_misalignment(image_path, preproc_out_dir):
    img = cv2_imread_jp(image_path)
    if img is None:
        return None, None, None, None
        
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # ---------------------------------------------------------
    # 1. 外径抽出（エッジ検出の前処理を強化）
    # ---------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    blurred = cv2.medianBlur(enhanced, 9)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_1_edges.jpg"), edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coin_cx, coin_cy = img.shape[1]//2, img.shape[0]//2
    coin_radius = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        coin_cx, coin_cy = int(x), int(y)
        coin_radius = int(radius)
        
    cv2.circle(color_img, (coin_cx, coin_cy), coin_radius, (255, 0, 0), 2)
    cv2.drawMarker(color_img, (coin_cx, coin_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # ---------------------------------------------------------
    # 2. 穴の二値化抽出
    # ---------------------------------------------------------
    roi_size = 120
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(img.shape[1], coin_cx + roi_size//2)
    y2 = min(img.shape[0], coin_cy + roi_size//2)
    
    roi = img[y1:y2, x1:x2]
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_2_roi_original.jpg"), roi)
    
    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (roi_size//2, roi_size//2), 40, 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_3_roi_thresh.jpg"), thresh)
    
    dark_pixels = cv2.countNonZero(thresh)
    is_hole = 1 if dark_pixels > 300 else 0
    
    is_misaligned = 0
    distance = 0.0
    
    if is_hole:
        M = cv2.moments(thresh)
        if M["m00"] != 0:
            hole_cx_roi = int(M["m10"] / M["m00"])
            hole_cy_roi = int(M["m01"] / M["m00"])
            
            hole_cx = x1 + hole_cx_roi
            hole_cy = y1 + hole_cy_roi
            
            distance = math.sqrt((hole_cx - coin_cx)**2 + (hole_cy - coin_cy)**2)
            
            if distance > 10.0:
                is_misaligned = 1
                
            cv2.drawMarker(color_img, (hole_cx, hole_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.line(color_img, (coin_cx, coin_cy), (hole_cx, hole_cy), (0, 255, 255), 2)
            
    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    hole_str = "HOLE OK" if is_hole else "NO HOLE"
    if is_hole and is_misaligned:
        hole_str = "HOLE MISALIGNED!"
        
    color = (0, 255, 0) if is_hole and not is_misaligned else (0, 0, 255)
    if not is_hole: color = (0, 255, 255)
    
    cv2.putText(color_img, f"{hole_str} (Dist:{distance:.1f})", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_4_final_result.jpg"), color_img)
        
    return is_hole, is_misaligned, dark_pixels, distance

def get_next_try_folder(base_dir):
    """次のtry_XXXフォルダ名を取得して作成する"""
    i = 1
    while True:
        folder_name = f"try_{i:03d}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1

if __name__ == "__main__":
    csv_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル_v2.csv"
    img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    base_out_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\50円玉写真の前処理"
    
    os.makedirs(base_out_dir, exist_ok=True)
    
    # 1. 新しいトライ用のフォルダを作成
    trial_dir = get_next_try_folder(base_out_dir)
    print(f"--- 50円玉 実画像 穴＆ズレ検出 [{os.path.basename(trial_dir)}] ---")
    
    # 2. この実行で使用したスクリプト自身をトライフォルダにコピー（履歴保存）
    current_script_path = os.path.abspath(__file__)
    shutil.copy(current_script_path, os.path.join(trial_dir, "evaluate_script.py"))
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    results = [] # 評価結果を保存するためのリスト
    
    for idx, row in df.iterrows():
        filename = row["ファイル名"]
        true_hole = row["中心穴の有無(1:あり/0:なし)"]
        true_misalign = row["穴のズレ(1:ズレあり/0:正常)"]
        
        img_path = os.path.join(img_dir, filename)
        
        # 処理を実行し、画像をトライフォルダに出力
        pred_hole, pred_misalign, score, dist = detect_center_hole_and_misalignment(img_path, trial_dir)
        
        if pred_hole is None:
            continue
            
        # 判定
        ok_hole = (pred_hole == true_hole)
        ok_misalign = (pred_misalign == true_misalign) if true_hole == 1 else True # 穴なしの時はズレ判定はスルー
        
        mark = "✅" if (ok_hole and ok_misalign) else "❌"
        
        print(f"[{mark}] {filename}")
        
        # 結果を記録
        results.append({
            "ファイル名": filename,
            "正解_穴有無": true_hole,
            "予測_穴有無": pred_hole,
            "判定_穴有無": "OK" if ok_hole else "NG",
            "面積スコア": score,
            "正解_穴ズレ": true_misalign,
            "予測_穴ズレ": pred_misalign,
            "判定_穴ズレ": "OK" if ok_misalign else "NG",
            "ズレ距離(px)": round(dist, 1),
            "総合判定": "OK" if (ok_hole and ok_misalign) else "NG"
        })
        
    # 3. 評価結果をCSVとしてトライフォルダに保存
    results_df = pd.DataFrame(results)
    csv_out_path = os.path.join(trial_dir, "evaluation_results.csv")
    results_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig") # Excelで文字化けしないBOM付きUTF-8
    
    # 精度計算
    hole_acc = (results_df["判定_穴有無"] == "OK").mean() * 100
    misalign_acc = (results_df["判定_穴ズレ"] == "OK").mean() * 100
    
    print(f"\n✅ 処理完了！")
    print(f"- 穴の有無 精度: {hole_acc:.1f}%")
    print(f"- 穴のズレ 精度: {misalign_acc:.1f}%")
    print(f"\n以下のフォルダに「出力画像」「実行スクリプト」「評価結果CSV」の全てを保存しました。")
    print(f"-> {trial_dir}")
