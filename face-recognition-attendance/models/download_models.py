import os
import urllib.request
import sys

# Directory to save the models
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# Model details
MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def download_progress(block_num, block_size, total_size):
    """Callback function to display download progress"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded * 100) / total_size)
        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB)")
    else:
        sys.stdout.write(f"\rDownloading... ({downloaded / (1024*1024):.2f}MB downloaded)")
    sys.stdout.flush()

def download_models():
    print("Starting download of face recognition models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for filename, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100 * 1024:
            print(f"Model already exists locally: {filename} ({os.path.getsize(filepath) / (1024*1024):.2f} MB). Skipping download.")
            continue
            
        print(f"\nDownloading {filename} from {url}...")
        try:
            # Custom User-Agent to prevent blockages
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                
                with open(filepath, 'wb') as out_file:
                    block_size = 1024 * 64
                    block_num = 0
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        out_file.write(block)
                        block_num += 1
                        download_progress(block_num, block_size, total_size)
            print(f"\nSuccessfully downloaded {filename}!")
        except Exception as e:
            print(f"\nError downloading {filename}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            sys.exit(1)

if __name__ == "__main__":
    download_models()
