"""
50円玉 穴＆ズレ検出 - ゼロベース再設計版
============================================
設計思想:
  - 基準は「外径（金属円盤の最外径）」の中心
  - 外枠も穴も、両方とも HoughCircles（ハフ変換）で円として検出する
  - 画像サイズが225px〜640pxまでバラバラなので、全パラメータを画像サイズに適応させる
  - 穴のない旧型50円は「穴の円が見つからない」で自然に0判定

パイプライン構成:
  Step0: 画像読み込み＋前処理（CLAHE＋メディアンフィルタ）
  Step1: HoughCircles で外枠（大きな円）を検出
  Step2: HoughCircles で穴（小さな円）を検出
  Step3: 穴の有無判定 = 穴の円が見つかったか
  Step4: 穴のズレ判定 = 外枠中心と穴中心の距離 / 外枠半径 > 閾値
"""

import cv2
import numpy as np
import os
import pandas as pd
import math
import shutil


def cv2_imread_jp(file_path):
    """日本語パスに対応した画像読み込み"""
    nparr = np.fromfile(file_path, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


def cv2_imwrite_jp(file_path, img):
    """日本語パスに対応した画像書き出し"""
    ext = os.path.splitext(file_path)[1]
    result, n = cv2.imencode(ext, img)
    if result:
        with open(file_path, mode='w+b') as f:
            n.tofile(f)


# =========================================================================
# Step1: 外枠（コインの最外径）を HoughCircles で検出
# =========================================================================
def detect_coin_outer(gray, debug_name=""):
    """コインの外枠を検出し、(cx, cy, radius) を返す。
    
    50円玉の外径は21.0mm。画像中でコインは画面の50〜95%程度を占める。
    画像サイズに応じて探索半径範囲を自動計算する。
    """
    h, w = gray.shape
    min_dim = min(h, w)
    
    # コインの半径は画像の短辺の20%〜45%の範囲に収まるはず
    min_r = max(20, int(min_dim * 0.20))
    max_r = int(min_dim * 0.45)  # 0.50→0.45に: 外枠が背景までハミ出すのを防ぐ
    
    # 前処理: コントラスト強調 + ノイズ除去
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 9)
    
    # 複数のparam2（感度）で試し、最も妥当な円を選ぶ
    all_candidates = []
    
    for param2 in [60, 50, 40, 30, 20]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 2,
            param1=100,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r
        )
        if circles is not None:
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                all_candidates.append((cx, cy, r, param2))
            if len(all_candidates) >= 3:
                break
    
    if not all_candidates:
        return w // 2, h // 2, min_dim // 3
    
    # 「エッジの強さ」で円の正確さを検証する
    # 検出された円の円周上のピクセルについて、エッジ（Canny）が通っている割合を計算
    edges = cv2.Canny(blurred, 50, 120)
    best_circle = None
    best_score = -1
    
    for cx, cy, r, p2 in all_candidates:
        # 円周上の点を36点サンプリング
        edge_count = 0
        total = 0
        for angle_deg in range(0, 360, 10):
            rad = math.radians(angle_deg)
            px = int(cx + r * math.cos(rad))
            py = int(cy + r * math.sin(rad))
            if 0 <= px < w and 0 <= py < h:
                total += 1
                # 円周付近±3pxにエッジがあるか
                for dr in range(-3, 4):
                    px2 = int(cx + (r + dr) * math.cos(rad))
                    py2 = int(cy + (r + dr) * math.sin(rad))
                    if 0 <= px2 < w and 0 <= py2 < h and edges[py2, px2] > 0:
                        edge_count += 1
                        break
        
        edge_ratio = edge_count / total if total > 0 else 0
        # 画像中心に近いほどボーナス
        dist_to_center = math.sqrt((cx - w/2)**2 + (cy - h/2)**2)
        center_score = max(0, 1.0 - (dist_to_center / (min_dim * 0.5)))
        
        score = edge_ratio * 0.7 + center_score * 0.3
        if score > best_score:
            best_score = score
            best_circle = (cx, cy, r)
    
    return best_circle


# =========================================================================
# Step2: 穴（コインの中心付近の小さな円）を HoughCircles で検出
# =========================================================================
def detect_hole(gray, coin_cx, coin_cy, coin_radius, debug_name=""):
    """コインの中心付近にある穴を HoughCircles で検出し、暗さ検証で確認する。
    
    50円玉の穴径は4.0mm、外径21.0mm → 穴の半径は外径半径の約19%
    
    ★重要: HoughCirclesだけでは「穴のない旧型50円」の中心リング影を
    穴と誤検出してしまう。そこで検出した円の内部が本当に「暗い＝貫通穴」
    かどうかを、周囲のピクセル平均値との差で検証する。
    
    Returns: (hole_found, hole_cx, hole_cy, hole_radius)
    """
    h, w = gray.shape
    
    # ROI: コインの中心から半径の70%の範囲を切り出す
    roi_half = int(coin_radius * 0.7)
    x1 = max(0, coin_cx - roi_half)
    y1 = max(0, coin_cy - roi_half)
    x2 = min(w, coin_cx + roi_half)
    y2 = min(h, coin_cy + roi_half)
    roi = gray[y1:y2, x1:x2]
    
    if roi.size == 0:
        return False, 0, 0, 0
    
    # 穴の半径の推定範囲（外枠半径の8%〜30%）
    hole_min_r = max(3, int(coin_radius * 0.08))
    hole_max_r = int(coin_radius * 0.30)
    
    # 前処理
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(roi)
    blurred = cv2.medianBlur(enhanced, 5)
    
    candidates = []
    
    for param2 in [40, 30, 20, 15, 10]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=roi.shape[0] // 3,
            param1=80,
            param2=param2,
            minRadius=hole_min_r,
            maxRadius=hole_max_r
        )
        if circles is not None:
            roi_cx, roi_cy = roi.shape[1] // 2, roi.shape[0] // 2
            for c in circles[0]:
                hx, hy, hr = int(c[0]), int(c[1]), int(c[2])
                # コイン中心から穴半径の50%以上離れたものは除外
                dist = math.sqrt((hx - roi_cx)**2 + (hy - roi_cy)**2)
                max_allowed = coin_radius * 0.5
                if dist > max_allowed:
                    continue
                candidates.append((x1 + hx, y1 + hy, hr, dist))
            if len(candidates) >= 3:
                break
    
    if not candidates:
        return False, 0, 0, 0
    
    # ★暗さ検証: 穴候補の内部が本当に暗い（=貫通穴）かどうかを確認
    # 「穴の内部の平均輝度」 vs 「穴の周囲リング（外側2倍半径）の平均輝度」を比較
    best_hole = None
    best_darkness = 0  # 暗さの差が大きいほど「本物の穴」
    
    for hcx, hcy, hr, dist in candidates:
        # 穴の内部のマスク
        hole_mask = np.zeros_like(gray)
        cv2.circle(hole_mask, (hcx, hcy), max(1, hr - 2), 255, -1)
        hole_pixels = gray[hole_mask > 0]
        
        # 穴の周囲リングのマスク（穴半径の1.5〜2.5倍のリング領域）
        ring_mask = np.zeros_like(gray)
        cv2.circle(ring_mask, (hcx, hcy), int(hr * 2.5), 255, -1)
        cv2.circle(ring_mask, (hcx, hcy), int(hr * 1.5), 0, -1)
        ring_pixels = gray[ring_mask > 0]
        
        if len(hole_pixels) < 5 or len(ring_pixels) < 5:
            continue
        
        hole_mean = np.mean(hole_pixels)
        ring_mean = np.mean(ring_pixels)
        darkness_diff = ring_mean - hole_mean  # 正の値=穴が暗い=本物の穴
        
        # 穴の内部は周囲より少なくとも20ポイント暗くなければ「偽の穴」
        if darkness_diff > 20 and darkness_diff > best_darkness:
            best_darkness = darkness_diff
            best_hole = (hcx, hcy, hr)
    
    if best_hole is None:
        return False, 0, 0, 0
    
    return True, best_hole[0], best_hole[1], best_hole[2]


# =========================================================================
# Step3 + Step4: 穴の有無判定 ＋ 穴のズレ判定
# =========================================================================
def judge_hole_and_misalignment(coin_cx, coin_cy, coin_radius,
                                 hole_found, hole_cx, hole_cy, hole_radius):
    """
    穴の有無: hole_found がTrue/Falseで決定
    穴のズレ: 外枠中心と穴中心のユークリッド距離を、外枠半径で正規化して判定
    
    ズレ判定の閾値: 外枠半径の3%以上ズレていたら「ズレあり」
    （50円玉の外径21mm → 半径10.5mm → 3%は約0.3mm = かなり厳しめ）
    """
    pred_hole = 1 if hole_found else 0
    
    if not hole_found:
        return pred_hole, 0, 0.0, 0.0
    
    distance = math.sqrt((hole_cx - coin_cx)**2 + (hole_cy - coin_cy)**2)
    ratio = distance / coin_radius if coin_radius > 0 else 0.0
    
    # ズレ判定の閾値（外枠半径の5%）
    MISALIGN_THRESHOLD = 0.05
    pred_misalign = 1 if ratio > MISALIGN_THRESHOLD else 0
    
    return pred_hole, pred_misalign, distance, ratio


# =========================================================================
# デバッグ画像の生成
# =========================================================================
def save_debug_image(gray, coin_cx, coin_cy, coin_radius,
                     hole_found, hole_cx, hole_cy, hole_radius,
                     pred_hole, pred_misalign, distance, ratio,
                     base_filename, out_dir):
    """結果を可視化したデバッグ画像を保存"""
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 外枠（緑）
    cv2.circle(color, (coin_cx, coin_cy), coin_radius, (0, 255, 0), 2)
    cv2.drawMarker(color, (coin_cx, coin_cy), (0, 255, 0),
                   cv2.MARKER_CROSS, 20, 2)
    
    if hole_found:
        # 穴（赤）
        cv2.circle(color, (hole_cx, hole_cy), hole_radius, (0, 0, 255), 2)
        cv2.drawMarker(color, (hole_cx, hole_cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 15, 2)
        # 距離の線（黄色）
        cv2.line(color, (coin_cx, coin_cy), (hole_cx, hole_cy), (0, 255, 255), 2)
        
        label = f"Dist:{distance:.1f}px Ratio:{ratio:.3f} Misalign:{pred_misalign}"
    else:
        label = "NO HOLE DETECTED"
    
    cv2.putText(color, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2_imwrite_jp(os.path.join(out_dir, f"{base_filename}_result.jpg"), color)


# =========================================================================
# 実験管理
# =========================================================================
def get_next_try_folder(base_dir):
    i = 1
    while True:
        folder_name = f"try_{i:03d}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1


# =========================================================================
# メイン処理
# =========================================================================
if __name__ == "__main__":
    csv_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル.csv"
    img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    base_out_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\50円玉写真の前処理"
    
    os.makedirs(base_out_dir, exist_ok=True)
    trial_dir = get_next_try_folder(base_out_dir)
    trial_name = os.path.basename(trial_dir)
    
    print(f"--- 50円玉 穴＆ズレ検出 [{trial_name}] ---")
    print(f"  ★ゼロベース再設計: 外枠も穴もHoughCirclesで統一検出")
    print(f"  ★基準: 外径（金属円盤の最外径）の中心")
    
    # スクリプトをコピーして実験を再現可能にする
    shutil.copy(os.path.abspath(__file__), os.path.join(trial_dir, "evaluate_script.py"))
    
    # 正解ラベル読み込み
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    results = []
    
    for idx, row in df.iterrows():
        filename = str(row["ファイル名"]).strip()
        true_hole = int(row["中心穴の有無(1:あり/0:なし)"])
        true_misalign = int(row["穴のズレ(1:ズレあり/0:正常)"])
        
        img_path = os.path.join(img_dir, filename)
        gray = cv2_imread_jp(img_path)
        if gray is None:
            print(f"[SKIP] {filename} - 読み込み失敗")
            continue
        
        base_filename = os.path.splitext(filename)[0]
        
        # Step1: 外枠検出
        coin_cx, coin_cy, coin_radius = detect_coin_outer(gray, base_filename)
        
        # Step2: 穴検出
        hole_found, hole_cx, hole_cy, hole_radius = detect_hole(
            gray, coin_cx, coin_cy, coin_radius, base_filename)
        
        # Step3+4: 判定
        pred_hole, pred_misalign, distance, ratio = judge_hole_and_misalignment(
            coin_cx, coin_cy, coin_radius,
            hole_found, hole_cx, hole_cy, hole_radius)
        
        # デバッグ画像保存
        save_debug_image(gray, coin_cx, coin_cy, coin_radius,
                         hole_found, hole_cx, hole_cy, hole_radius,
                         pred_hole, pred_misalign, distance, ratio,
                         base_filename, trial_dir)
        
        # 精度チェック
        ok_hole = (pred_hole == true_hole)
        ok_misalign = (pred_misalign == true_misalign) if true_hole == 1 else True
        
        mark = "✅" if (ok_hole and ok_misalign) else "❌"
        print(f"[{mark}] {filename:25s} | 穴: {true_hole}/{pred_hole} "
              f"| ズレ: {true_misalign}/{pred_misalign} "
              f"(dist:{distance:.1f}px ratio:{ratio:.3f} r:{coin_radius})")
        
        results.append({
            "ファイル名": filename,
            "正解_穴有無": true_hole,
            "予測_穴有無": pred_hole,
            "判定_穴有無": "OK" if ok_hole else "NG",
            "正解_穴ズレ": true_misalign,
            "予測_穴ズレ": pred_misalign,
            "判定_穴ズレ": "OK" if ok_misalign else "NG",
            "ズレ距離(px)": round(distance, 1),
            "ズレ比率": round(ratio, 4),
            "コイン半径(px)": coin_radius,
            "穴半径(px)": hole_radius if hole_found else 0,
            "総合判定": "OK" if (ok_hole and ok_misalign) else "NG"
        })
    
    # 結果出力
    results_df = pd.DataFrame(results)
    csv_out = os.path.join(trial_dir, "evaluation_results.csv")
    excel_out = os.path.join(trial_dir, "evaluation_results.xlsx")
    results_df.to_csv(csv_out, index=False, encoding="utf-8-sig", errors="ignore")
    results_df.to_excel(excel_out, index=False)
    
    hole_acc = (results_df["判定_穴有無"] == "OK").mean() * 100
    misalign_acc = (results_df["判定_穴ズレ"] == "OK").mean() * 100
    total_acc = (results_df["総合判定"] == "OK").mean() * 100
    
    print(f"\n✅ 処理完了！")
    print(f"- パイプラインA【穴の有無】精度: {hole_acc:.1f}%")
    print(f"- パイプラインB【穴のズレ】精度: {misalign_acc:.1f}%")
    print(f"- 総合判定 精度: {total_acc:.1f}%")
    print(f"-> {trial_dir}")
