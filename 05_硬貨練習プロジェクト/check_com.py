import cv2
import numpy as np
import os
import math

base_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\05_硬貨練習プロジェクト\実験結果\50円玉写真の前処理\try_021"
img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"

def check_engraving_center(filename):
    nparr = np.fromfile(os.path.join(img_dir, filename), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    blurred = cv2.medianBlur(img, 9)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=50, minRadius=100, maxRadius=450
    )
    if circles is None: return
    
    c = circles[0][0]
    coin_cx, coin_cy, coin_radius = int(c[0]), int(c[1]), int(c[2])
    
    # 穴の中心を絶対閾値＋ブラックハットで出す
    roi_size = int(coin_radius * 0.6)
    x1 = max(0, coin_cx - roi_size//2)
    y1 = max(0, coin_cy - roi_size//2)
    x2 = min(img.shape[1], coin_cx + roi_size//2)
    y2 = min(img.shape[0], coin_cy + roi_size//2)
    roi = img[y1:y2, x1:x2]
    _, thresh_abs = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    k_size = int(coin_radius * 0.6)
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
    _, thresh_bh = cv2.threshold(bh, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_or(thresh_abs, thresh_bh)
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (thresh.shape[1]//2, thresh.shape[0]//2), int(coin_radius * 0.4), 255, -1)
    thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    best_hole = max(contours, key=cv2.contourArea)
    (hx, hy), _ = cv2.minEnclosingCircle(best_hole)
    hole_cx = x1 + int(hx)
    hole_cy = y1 + int(hy)
    
    # 硬貨全体（穴を除く）の重心を計算する
    # 硬貨の外枠だけを切り出す
    coin_mask = np.zeros_like(img)
    cv2.circle(coin_mask, (coin_cx, coin_cy), coin_radius, 255, -1)
    # 穴の部分は除外する
    cv2.circle(coin_mask, (hole_cx, hole_cy), int(coin_radius * 0.25), 0, -1)
    
    coin_only = cv2.bitwise_and(img, coin_mask)
    # 全体を二値化して「刻印の暗い部分」を抽出
    _, thresh_engraving = cv2.threshold(coin_only, 100, 255, cv2.THRESH_BINARY_INV)
    thresh_engraving = cv2.bitwise_and(thresh_engraving, coin_mask)
    
    M = cv2.moments(thresh_engraving)
    if M["m00"] != 0:
        eng_cx = int(M["m10"] / M["m00"])
        eng_cy = int(M["m01"] / M["m00"])
        dist_hole_to_coin = math.sqrt((hole_cx - coin_cx)**2 + (hole_cy - coin_cy)**2)
        dist_eng_to_coin = math.sqrt((eng_cx - coin_cx)**2 + (eng_cy - coin_cy)**2)
        dist_hole_to_eng = math.sqrt((hole_cx - eng_cx)**2 + (hole_cy - eng_cy)**2)
        
        print(f"{filename:15} | Hole-Outer: {dist_hole_to_coin:5.1f} | Eng-Outer: {dist_eng_to_coin:5.1f} | Hole-Eng: {dist_hole_to_eng:5.1f}")

for f in ["50yen_001.jpeg", "50yen_007.jpeg", "50yen_008.jpeg", "50yen_010.jpeg"]:
    check_engraving_center(f)
