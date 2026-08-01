import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print("AI学習に必要なライブラリをインストールしています...")
    # YOLOv8のためのUltralytics
    install("ultralytics")
    # OpenCV（画像処理用）
    install("opencv-python")
    print("インストールが完了しました！これでAIを動かす準備が整いました。")
