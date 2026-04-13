import os
import json
import glob
import shutil
import random
import cv2

def convert_to_yolo_file(dirpath):
    """Parse JSON annotations and convert them into standard YOLO format (.txt)."""
    label_dir = os.path.join(dirpath, 'labels')
    if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
        print(f'{label_dir} already exists.')
        return
        
    os.makedirs(label_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(dirpath, '*.json'))

    for json_file in json_files:
        file_id = os.path.basename(json_file).replace('.json', '')
        img_path = os.path.join(dirpath, f'{file_id}.jpg')

        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        with open(json_file, 'r') as f:
            data = json.load(f)

        lines = []
        for marker in data['markers']:
            corner = marker['corners']
            xs = [c[0] for c in corner]
            ys = [c[1] for c in corner]
                
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_mid_norm = ((x_min + x_max) / 2) / width
            y_mid_norm = ((y_min + y_max) / 2) / height
            w_norm = (x_max - x_min) / width
            h_norm = (y_max - y_min) / height

            line = f'0 {x_mid_norm:.6f} {y_mid_norm:.6f} {w_norm:.6f} {h_norm:.6f}'
            lines.append(line)

        txt_file = os.path.join(label_dir, f'{file_id}.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(lines))
    print("Convert to YOLO format successfully.")

def split_dataset(src_dir, base_dir="data/processed/dataset", train_ratio=0.8):
    """Split data into train/val sets and organize directory structure."""
    sub_dirs = ["images/train", "images/val", "labels/train", "labels/val"]

    for sub in sub_dirs:
        try:
            os.makedirs(os.path.join(base_dir, sub))
        except FileExistsError:
            pass
        
    all_ids = [f.replace('.jpg', '') for f in os.listdir(src_dir) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_ids)

    split_index = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split_index]
    val_ids = all_ids[split_index:]

    def move_files(ids, split_name):
        label_src_dir = os.path.join(src_dir, "labels")
        for file_id in ids:
            img_src = os.path.join(src_dir, f"{file_id}.jpg")
            img_dst = os.path.join(base_dir, f"images/{split_name}/{file_id}.jpg")
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)

            txt_src = os.path.join(label_src_dir, f"{file_id}.txt")
            txt_dst = os.path.join(base_dir, f"labels/{split_name}/{file_id}.txt")
            if os.path.exists(txt_src):
                shutil.copy(txt_src, txt_dst)

    move_files(train_ids, "train")
    move_files(val_ids, "val")
    print("Dataset split successfully.")

if __name__ == "__main__":
    RAW_DIR = "data/raw/flyingarucov2"
    convert_to_yolo_file(RAW_DIR)
    split_dataset(RAW_DIR)