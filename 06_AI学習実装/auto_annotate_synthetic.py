import cv2
import numpy as np
import os
import random
import math

# --- 1. ベース画像と初期アノテーション（人間が1回だけ設定する） ---
# ここでは仮の座標を使用します（本番環境では最初の1枚だけ手作業で枠を描きます）
BASE_IMG_PATH = r"c:\__0_kohei_desktop\01_機械設計\03_刻印読む\03_データ探し\50円玉写真\50yen_001.jpeg"

# 仮のバウンディングボックス [x, y, w, h]
# 50円玉(001.jpeg)の中心座標を (300, 240) と仮定し、その上の「5」と「0」の位置を推測
BBOX_5 = [200, 70, 70, 80]  # 左側の「5」
BBOX_0 = [300, 70, 80, 90]  # 右側の「0」

def get_bbox_corners(bbox):
    x, y, w, h = bbox
    return np.array([
        [x, y, 1],
        [x+w, y, 1],
        [x+w, y+h, 1],
        [x, y+h, 1]
    ]).T

def create_synthetic_data(num_samples=10):
    output_dir = "runs/synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(BASE_IMG_PATH, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    base_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if base_img is None:
        print("ベース画像が読み込めませんでした。")
        return
        
    h, w = base_img.shape[:2]
    cx, cy = w / 2, h / 2
    
    # 元画像に仮のアノテーション枠を描画したものを保存（確認用）
    debug_base = base_img.copy()
    cv2.rectangle(debug_base, (BBOX_5[0], BBOX_5[1]), (BBOX_5[0]+BBOX_5[2], BBOX_5[1]+BBOX_5[3]), (0, 0, 255), 2)
    cv2.rectangle(debug_base, (BBOX_0[0], BBOX_0[1]), (BBOX_0[0]+BBOX_0[2], BBOX_0[1]+BBOX_0[3]), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "00_base_annotated.jpg"), debug_base)
    
    print(f"🔄 {num_samples}枚の合成データとYOLOラベルを自動生成します...")
    
    for i in range(num_samples):
        # 1. ランダムな変換パラメータの決定
        angle = random.uniform(0, 360)      # 0〜360度のランダム回転
        tx = random.uniform(-50, 50)        # X方向のランダム平行移動
        ty = random.uniform(-50, 50)        # Y方向のランダム平行移動
        scale = random.uniform(0.9, 1.1)    # 90%〜110%のランダム拡大縮小
        
        # 2. 画像のアフィン変換行列を作成
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        # 平行移動を追加
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        # 画像を変形
        synth_img = cv2.warpAffine(base_img, M_rot, (w, h), borderValue=(0,0,0))
        
        # 3. アノテーション枠の座標変換（数学的計算）
        corners_5 = get_bbox_corners(BBOX_5)
        corners_0 = get_bbox_corners(BBOX_0)
        
        new_corners_5 = M_rot.dot(corners_5).T
        new_corners_0 = M_rot.dot(corners_0).T
        
        # 変換後の4頂点から新しいバウンディングボックス（外接矩形）を計算
        x5, y5, w5, h5 = cv2.boundingRect(np.float32(new_corners_5))
        x0, y0, w0, h0 = cv2.boundingRect(np.float32(new_corners_0))
        
        # 4. YOLO形式のラベル出力 (class_id x_center y_center width height) は省略し、今回は確認用の画像を描画
        debug_synth = synth_img.copy()
        cv2.rectangle(debug_synth, (x5, y5), (x5+w5, y5+h5), (0, 0, 255), 2)
        cv2.putText(debug_synth, "5", (x5, y5-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.rectangle(debug_synth, (x0, y0), (x0+w0, y0+h0), (255, 0, 0), 2)
        cv2.putText(debug_synth, "0", (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        file_name = f"synth_{i+1:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, file_name), debug_synth)
        
    print(f"✅ 生成完了！ 結果は {output_dir} を確認してください。")

if __name__ == "__main__":
    create_synthetic_data(10)
