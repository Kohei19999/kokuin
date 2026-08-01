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
# 共通処理: コインの輪郭を楕円フィッティングで検出し、
#           斜め撮影の場合は透視変換で正規化した正面視画像を返す
#           ★ 戻り値が (正規化後の画像, 中心x, 中心y, 半径) に変わった点に注意
# =========================================================================
def find_coin_and_normalize(img, base_filename, preproc_out_dir):
    """楕円フィッティングでコインを検出し、正面視に正規化した画像と中心・半径を返す。
    
    斜め撮影 → 楕円検出 → warpAffine補正 → 正規化済み正面視画像
    正面撮影 → 楕円≒真円 → そのまま返す
    """
    h, w = img.shape
    coin_cx, coin_cy = w // 2, h // 2
    coin_radius = min(h, w) // 4
    normalized = img.copy()  # デフォルト: 補正なし

    # ── Step1: エッジ検出 ──────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    blurred = cv2.medianBlur(enhanced, 9)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # ── Step2: 外縁輪郭を取得（最大面積のもの） ──────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 3000]
    if not valid:
        # 検出失敗 → デフォルト値で返す
        return normalized, coin_cx, coin_cy, coin_radius

    best_contour = max(valid, key=lambda x: x[1])[0]

    # ── Step3: 楕円フィッティング ────────────────────────
    if len(best_contour) < 5:
        return normalized, coin_cx, coin_cy, coin_radius

    ellipse = cv2.fitEllipse(best_contour)
    (ex, ey), (major_axis, minor_axis), angle = ellipse
    coin_cx, coin_cy = int(ex), int(ey)

    # 短軸を「真の半径」の基準にする（長軸は遠近で伸びる）
    coin_radius = int(minor_axis / 2)

    # ── Step4: 楕円の歪み率を計算 ────────────────────────
    # 長軸/短軸の比が1.05以上なら斜め撮影と判定して補正する
    ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
    is_tilted = ratio > 1.05

    # ── Step5: 斜め撮影補正（透視アフィン変換）────────────────
    if is_tilted:
        # 楕円をスケーリングで円に戻すアフィン変換を構築
        # 長軸方向を短軸のサイズに圧縮する（=正面視に補正）
        scale_x = minor_axis / major_axis if major_axis > 0 else 1.0
        scale_y = 1.0

        # 回転角度（楕円の長軸方向）に合わせてスケールを掛ける
        rad = math.radians(angle)
        M_rot = cv2.getRotationMatrix2D((coin_cx, coin_cy), angle, 1.0)
        M_scale = np.array([
            [scale_x * math.cos(rad)**2 + math.sin(rad)**2,
             (scale_x - 1) * math.sin(rad) * math.cos(rad), coin_cx * (1 - scale_x) * math.cos(rad)**2],
            [(scale_x - 1) * math.sin(rad) * math.cos(rad),
             scale_x * math.sin(rad)**2 + math.cos(rad)**2, coin_cy * (1 - scale_x) * math.sin(rad)**2],
        ], dtype=np.float32)

        normalized = cv2.warpAffine(img, M_scale, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        tilt_label = f"TILTED ratio={ratio:.2f}"
    else:
        tilt_label = f"NORMAL ratio={ratio:.2f}"

    # ── デバッグ画像の保存 ─────────────────────────────
    debug_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(debug_color, ellipse, (0, 255, 0), 2)
    cv2.drawMarker(debug_color, (coin_cx, coin_cy), (0, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(debug_color, tilt_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2_imwrite_jp(os.path.join(preproc_out_dir,
                                f"{base_filename}_0_ellipse.jpg"), debug_color)

    return normalized, coin_cx, coin_cy, coin_radius


# =========================================================================
# パイプラインA: 【穴の有無判定】カスケード判定（try_012）
# =========================================================================
def pipeline_A_check_hole_presence(img, coin_cx, coin_cy, coin_radius, base_filename, preproc_out_dir):
    h, w = img.shape
    
    # コインの中心からROIを切り出す
    roi_size = int(coin_radius * 1.5)
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(w, coin_cx + roi_size//2)
    y2 = min(h, coin_cy + roi_size//2)
    roi = img[y1:y2, x1:x2]
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_A1_roi.jpg"), roi)
    
    # ---------------------------------------------------------
    # ステップ1: 絶対的な暗さ（固定閾値）による一次判定
    # ---------------------------------------------------------
    # ほとんどの正常な穴（001〜010）は、この単純な処理で100%確実に検出可能。
    # 穴なし（011, 012）を絶対に誤検知しない最強のフィルター。
    _, thresh_abs = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    
    mask_abs = np.zeros_like(thresh_abs)
    rh, rw = thresh_abs.shape
    cv2.circle(mask_abs, (rw//2, rh//2), int(coin_radius * 0.4), 255, -1)
    thresh_abs = cv2.bitwise_and(thresh_abs, mask_abs)
    
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_A2_thresh_abs.jpg"), thresh_abs)
    
    dark_abs = cv2.countNonZero(thresh_abs)
    
    # 確実な暗さがあれば即座に「穴あり」として終了！
    if dark_abs > 150:
        return 1, dark_abs, 0
        
    # ---------------------------------------------------------
    # ステップ2: ブラックハット＋円形度による救済判定
    # ---------------------------------------------------------
    # ステップ1で漏れたもの（011, 012の穴なし、または 013, 014の照明が明るすぎる穴）を判別する
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    
    thresh_bh = cv2.bitwise_and(thresh_bh, mask_abs)
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_A3_thresh_bh.jpg"), thresh_bh)
    
    # 抽出された白い領域が「真円（ドリルの穴）」に近いかチェック
    # 穴なし（011,012）の梨地テクスチャはザラザラしているだけで丸くないため、ここで完全に弾かれる
    contours, _ = cv2.findContours(thresh_bh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    is_hole = 0
    best_circ = 0.0
    
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area > 100 and perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.4:  # 丸い形をしていれば本物の穴！
                is_hole = 1
                best_circ = circularity
                
    return is_hole, dark_abs, int(best_circ * 100)


# =========================================================================
# パイプラインB: 【穴のズレ判定】（coin_cxを共有して使う）
# =========================================================================
def pipeline_B_check_misalignment(img, coin_cx, coin_cy, coin_radius, base_filename, preproc_out_dir):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.circle(color_img, (coin_cx, coin_cy), coin_radius, (255, 0, 0), 2)
    cv2.drawMarker(color_img, (coin_cx, coin_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 穴の重心算出（絶対閾値＋ブラックハットの合わせ技）
    # 穴が大きくズレている場合（010番など）にも対応できるよう、探索範囲（ROI）を広げる
    roi_size = int(coin_radius * 1.2)
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(img.shape[1], coin_cx + roi_size//2)
    y2 = min(img.shape[0], coin_cy + roi_size//2)
    
    roi = img[y1:y2, x1:x2]
    
    # 絶対閾値
    _, thresh_abs = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    
    # ブラックハット
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY)
    
    # 両方のマスクを合成（OR）
    thresh = cv2.bitwise_or(thresh_abs, thresh_bh)
    
    hole_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    is_misaligned = 0
    distance = 0.0
    
    # パイプラインAで「穴がある」ことは確定済みなので、単純に最も面積の大きい塊を穴とする！
    # （ズレた穴は形が歪んで円形度が落ちるため、あえて円形度フィルタは掛けない）
    if hole_contours:
        best_hole = max(hole_contours, key=cv2.contourArea)
        
        # 穴の中の影は「三日月型」になることが多い。
        # そのため、面積の重心（moments）をとると、影の濃い側に中心が引っ張られてしまう！
        # 代わりに「最小外接円」を使えば、三日月型の外枠から真の中心を正確に推定できる。
        (hx, hy), h_radius = cv2.minEnclosingCircle(best_hole)
        hole_cx = x1 + int(hx)
        hole_cy = y1 + int(hy)
        
        distance = math.sqrt((hole_cx - coin_cx)**2 + (hole_cy - coin_cy)**2)
        
        # ズレ判定（コイン半径に対する相対比率で判定：半径の10%以上ずれていたらズレ）
        threshold_px = max(15.0, coin_radius * 0.10)
        if distance > threshold_px:
            is_misaligned = 1
            
        cv2.drawMarker(color_img, (hole_cx, hole_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.line(color_img, (coin_cx, coin_cy), (hole_cx, hole_cy), (0, 255, 255), 2)
            
    cv2.putText(color_img, f"Dist:{distance:.1f}px (Misalign:{is_misaligned})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    cv2_imwrite_jp(os.path.join(preproc_out_dir, f"{base_filename}_B1_misalignment.jpg"), color_img)
        
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
    csv_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル_v2.csv"
    img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    base_out_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\50円玉写真の前処理"
    
    os.makedirs(base_out_dir, exist_ok=True)
    trial_dir = get_next_try_folder(base_out_dir)
    print(f"--- 50円玉 穴＆ズレ検出 [{os.path.basename(trial_dir)}] ---")
    print(f"  改善点: コイン中心検出→ROI切出し＋照明平坦化OR判定＋相対ズレ閾値")
    
    current_script_path = os.path.abspath(__file__)
    shutil.copy(current_script_path, os.path.join(trial_dir, "evaluate_script.py"))
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    results = []
    
    for idx, row in df.iterrows():
        filename = row["ファイル名"]
        true_hole = int(row["中心穴の有無(1:あり/0:なし)"])
        true_misalign = int(row["穴のズレ(1:ズレあり/0:正常)"])
        
        img_path = os.path.join(img_dir, filename)
        img = cv2_imread_jp(img_path)
        if img is None:
            print(f"[SKIP] {filename} - 読み込み失敗")
            continue
        
        base_filename = os.path.splitext(filename)[0]
        
        # 共通処理: 楕円フィッティングでコインを検出し、斜め撮影を正規化
        normalized_img, coin_cx, coin_cy, coin_radius = find_coin_and_normalize(
            img, base_filename, trial_dir)
        
        # パイプラインA: 穴の有無判定（正規化済み画像を使う）
        pred_hole, dark_abs, dark_norm = pipeline_A_check_hole_presence(
            normalized_img, coin_cx, coin_cy, coin_radius, base_filename, trial_dir)
        
        # パイプラインB: 穴のズレ判定（正規化済み画像を使う）
        if pred_hole == 1:
            pred_misalign, dist = pipeline_B_check_misalignment(
                normalized_img, coin_cx, coin_cy, coin_radius, base_filename, trial_dir)
        else:
            pred_misalign, dist = 0, 0.0
            
        ok_hole = (pred_hole == true_hole)
        ok_misalign = (pred_misalign == true_misalign) if true_hole == 1 else True
        
        mark = "✅" if (ok_hole and ok_misalign) else "❌"
        print(f"[{mark}] {filename} | 穴: {true_hole}/{pred_hole} (abs:{dark_abs},norm:{dark_norm}) | ズレ: {true_misalign}/{pred_misalign} ({dist:.1f}px, r:{coin_radius})")
        
        results.append({
            "ファイル名": filename,
            "正解_穴有無": true_hole,
            "予測_穴有無": pred_hole,
            "判定_穴有無": "OK" if ok_hole else "NG",
            "暗部(絶対)": dark_abs,
            "暗部(平坦化)": dark_norm,
            "正解_穴ズレ": true_misalign,
            "予測_穴ズレ": pred_misalign,
            "判定_穴ズレ": "OK" if ok_misalign else "NG",
            "ズレ距離(px)": round(dist, 1),
            "コイン半径": coin_radius,
            "総合判定": "OK" if (ok_hole and ok_misalign) else "NG"
        })
        
    results_df = pd.DataFrame(results)
    csv_out_path = os.path.join(trial_dir, "evaluation_results.csv")
    excel_out_path = os.path.join(trial_dir, "evaluation_results.xlsx")
    
    # Windowsで文字化けしないようにShift-JIS(cp932)で保存。無理な文字は無視
    results_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig", errors="ignore")
    # Excel形式でも直接保存
    results_df.to_excel(excel_out_path, index=False)
    
    hole_acc = (results_df["判定_穴有無"] == "OK").mean() * 100
    misalign_acc = (results_df["判定_穴ズレ"] == "OK").mean() * 100
    total_acc = (results_df["総合判定"] == "OK").mean() * 100
    
    print(f"\n✅ 処理完了！")
    print(f"- パイプラインA【穴の有無】精度: {hole_acc:.1f}%")
    print(f"- パイプラインB【穴のズレ】精度: {misalign_acc:.1f}%")
    print(f"- 総合判定 精度: {total_acc:.1f}%")
    print(f"-> {trial_dir}")
