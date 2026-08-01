"""
新規追加画像（015-030）の一括可視化スクリプト
HoughCirclesでコイン検出を試みて、画像の多様性を把握する
"""
import cv2
import numpy as np
import os
import math

img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
out_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\preview_add"
os.makedirs(out_dir, exist_ok=True)

add_files = [f for f in sorted(os.listdir(img_dir)) if "_add" in f]

print(f"{'ファイル名':25} {'サイズ':>10} {'解像度':>15} {'ハフ円検出':>12}")
print("-" * 70)
for fname in add_files:
    fpath = os.path.join(img_dir, fname)
    nparr = np.fromfile(fpath, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"{fname:25} 読み込み失敗")
        continue
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    fsize = os.path.getsize(fpath)

    blurred = cv2.medianBlur(img_gray, 9)
    min_r = max(30, min(h, w) // 5)
    max_r = min(h, w) // 2
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=50, minRadius=min_r, maxRadius=max_r
    )

    canvas = img_color.copy()
    if circles is not None:
        c = circles[0][0]
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        cv2.circle(canvas, (cx, cy), r, (0, 255, 0), 3)
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        result = f"検出OK r={r}"
    else:
        result = "検出失敗"

    # サムネを保存（長辺を400pxに縮小）
    scale = 400 / max(h, w)
    thumb = cv2.resize(canvas, (int(w * scale), int(h * scale)))
    cv2.imwrite(os.path.join(out_dir, fname.replace(".jpeg", "_thumb.jpg")), thumb)

    print(f"{fname:25} {fsize:>8,}B  {w}x{h:>5}px  {result:>15}")
