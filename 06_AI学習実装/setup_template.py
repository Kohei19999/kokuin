import cv2
import numpy as np
import os

def create_template():
    # ほぼ水平に撮影されている 001 の画像を使用（データセット内の画像）
    img_path = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\dataset\train\images\50yen_001_jpeg.rf.9c78deac0a742b259cc21a5f66622a28.jpg"
    
    # 日本語パス対応の読み込み
    with open(img_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        print("画像が読み込めません。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    
    # ハフ変換でコインの中心を検出
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=100, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        
        # 50の文字は中心より上部（Yが小さい方向）にある
        # 半径rを基準に切り抜き座標を決定
        # 5と0の間の空間をしっかり含むように幅と高さを設定
        crop_w = int(r * 0.8)
        crop_h = int(r * 0.4)
        crop_x1 = int(cx - crop_w / 2)
        crop_y1 = int(cy - r * 0.75) # 中心より上
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        
        template = gray[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite("template_50.jpg", template)
        print("✅ テンプレート画像 'template_50.jpg' を作成しました！")
        
        # わかりやすいように枠を描画した確認用画像も保存
        cv2.rectangle(img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 2)
        cv2.imwrite("template_check.jpg", img)
    else:
        print("コインが検出できませんでした。")

if __name__ == "__main__":
    create_template()
