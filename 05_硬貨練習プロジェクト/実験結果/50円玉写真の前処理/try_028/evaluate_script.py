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
    """【最終ハイブリッド版】コインの外枠検出
    try_021で高精度だった「CLAHEなし＋MedianBlurのみ」のシンプルなHoughCirclesを採用。
    画像サイズに応じた動的スケーリングを適用し、画像中心付近の最も強い円を無条件で採用する。
    """
    h, w = gray.shape
    min_dim = min(h, w)
    
    # 50円玉のサイズ想定
    min_r = max(20, int(min_dim * 0.20))
    max_r = int(min_dim * 0.45)
    
    # 前処理: CLAHEなどのコントラスト強調は背景のノイズ（ノートの線など）を拾うので絶対にしない！
    blurred = cv2.medianBlur(gray, 9)
    
    best_circle = None
    
    # 感度を変えながら探索
    for param2 in [60, 50, 40, 30]:
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
            # HoughCirclesは「エッジが最も強い円」から順に返してくる
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # 画像の中心から極端に離れていないもの（背景ノイズ対策）を採用
                dist_to_center = math.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                if dist_to_center < min_dim * 0.3:
                    best_circle = (cx, cy, r)
                    break
            if best_circle is not None:
                break
                
    if best_circle is None:
        return w // 2, h // 2, min_dim // 3
        
    return best_circle


# =========================================================================
# Step2: 穴（コインの中心付近の小さな円）を HoughCircles で検出
# =========================================================================
def detect_hole(gray, coin_cx, coin_cy, coin_radius, debug_name=""):
    """【最終ハイブリッド版】貫通穴の検出
    try_012で100%の精度を出した「絶対的暗さ（漆黒の影）」による判定を復活させる。
    背景色が白でも黒でも、貫通穴の「切り口の側面」には必ず暗い影（<65）ができる物理法則を利用する。
    画像サイズに応じた動的な面積閾値を適用する。
    """
    h, w = gray.shape
    roi_size = int(coin_radius * 1.2)  # 中心がズレている場合も考慮し広めに取る
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(w, coin_cx + roi_size//2)
    y2 = min(h, coin_cy + roi_size//2)
    roi = gray[y1:y2, x1:x2]
    
    if roi.size == 0:
        return False, 0, 0, 0
        
    # 貫通穴の影（暗部）を絶対閾値で抽出
    _, thresh_abs = cv2.threshold(roi, 65, 255, cv2.THRESH_BINARY_INV)
    
    # 補助としてブラックハット（局所的な暗部）も抽出
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(blackhat, 60, 255, cv2.THRESH_BINARY)
    
    # 両方のマスクを合成（OR）
    mask = cv2.bitwise_or(thresh_abs, thresh_bh)
    
    # マスクの中から輪郭を探す
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 穴とみなすための最小面積（コイン半径に応じて動的変化: R=150なら約112px）
    min_area_req = (coin_radius ** 2) * 0.005 
    
    best_contour = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area and area > min_area_req:
            # 大きすぎるノイズ（背景など）は除外
            if area < (coin_radius ** 2) * 0.8:
                max_area = area
                best_contour = c
                
    if best_contour is None:
        return False, 0, 0, 0
        
    # 穴の中心と半径（外接円）を計算
    (hx, hy), hr = cv2.minEnclosingCircle(best_contour)
    return True, int(x1 + hx), int(y1 + hy), int(hr)


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
    csv_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉_正解ラベル_v2.csv"
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
