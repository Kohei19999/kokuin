import cv2
import numpy as np
import glob
import os

def check_contours():
    # 元画像のディレクトリ
    img_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真"
    image_paths = glob.glob(os.path.join(img_dir, "*.jp*g"))
    
    # 出力先ディレクトリ
    output_dir = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\06_AI学習実装\runs\contours_check"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        # 日本語パス対応読み込み
        with open(img_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        
        # ハフ変換で円を検出（外径）
        # 50円玉のサイズに合わせてパラメータを広めに設定
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=50, param2=30, minRadius=50, maxRadius=400)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 最初の円（一番強い検出結果）
            cx, cy, r = circles[0][0]
            
            # 外径を緑色で描画
            cv2.circle(img, (cx, cy), r, (0, 255, 0), 4)
            # 中心点を赤色で描画
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        
        # 保存
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, base_name)
        
        ext = os.path.splitext(out_path)[1]
        result, n = cv2.imencode(ext, img)
        if result:
            with open(out_path, mode='w+b') as f:
                n.tofile(f)
                
    print("✅ 輪郭抽出テスト完了！")

if __name__ == "__main__":
    check_contours()
