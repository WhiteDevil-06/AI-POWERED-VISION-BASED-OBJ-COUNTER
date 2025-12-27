import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
API_KEY = "9Dro6WbBZ9bW4iaU1Z53"  # Your Key
EPOCHS = 20
IMG_SIZE = 640
# Important: If her laptop has 6GB VRAM, Batch 16 is safe. If it crashes, change to 8.
BATCH_SIZE = 16  

def check_gpu():
    print("--- CHECKING FOR NVIDIA GPU ---")
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        return 0 # Use GPU 0
    else:
        print("⚠️ NO GPU DETECTED! Training will be SLOW (CPU Mode).")
        return 'cpu'

def setup_directories():
    print("\nStep 1: Setting up clean workspace...")
    if os.path.exists('mega_dataset'):
        shutil.rmtree('mega_dataset')
    
    os.makedirs('mega_dataset/train/images', exist_ok=True)
    os.makedirs('mega_dataset/train/labels', exist_ok=True)
    os.makedirs('mega_dataset/valid/images', exist_ok=True)
    os.makedirs('mega_dataset/valid/labels', exist_ok=True)

def download_data():
    print("\nStep 2: Downloading data from Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    # Dataset 1
    d1 = rf.workspace("project-wfbsj").project("cardboard-box-8uolq").version(1).download("yolov8")
    # Dataset 2
    d2 = rf.workspace("cardboard-box").project("cardboard-box-hql8b").version(1).download("yolov8")
    # Dataset 3 (Logistics)
    d3 = rf.workspace("roboflow-ngkro").project("logistics-h0uec").version(10).download("yolov8")
    
    return [d1, d2, d3]

def merge_datasets(datasets):
    print("\nStep 3: Merging datasets and fixing filenames...")
    
    for idx, dataset in enumerate(datasets):
        is_logistics = (idx == 2)
        source_folder = dataset.location
        folder_name = os.path.basename(source_folder)
        
        print(f"  -> Processing {folder_name} (Logistics Mode: {is_logistics})...")
        
        for split in ['train', 'valid']:
            src_img_dir = os.path.join(source_folder, split, 'images')
            src_lbl_dir = os.path.join(source_folder, split, 'labels')
            
            dst_img_dir = f"mega_dataset/{split}/images"
            dst_lbl_dir = f"mega_dataset/{split}/labels"
            
            if not os.path.exists(src_img_dir):
                continue
                
            files = os.listdir(src_img_dir)
            for i, f in enumerate(files):
                # 1. Copy Image with Short Name to fix "Long Path" errors
                name, ext = os.path.splitext(f)
                short_ptr = f"{folder_name}_{split}_{i:05d}"
                new_img_name = f"{short_ptr}{ext}"
                
                shutil.copy(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, new_img_name))
                
                # 2. Process Label
                txt_name = f"{name}.txt"
                new_txt_name = f"{short_ptr}.txt"
                src_txt_path = os.path.join(src_lbl_dir, txt_name)
                
                if os.path.exists(src_txt_path):
                    with open(src_txt_path, 'r') as file:
                        lines = file.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        cls = int(parts[0])
                        
                        # LOGISTICS FILTER: Class 2 is Box. Map to 0.
                        # OTHERS: Map Class X to 0.
                        if is_logistics:
                            if cls == 2:
                                new_lines.append(f"0 {' '.join(parts[1:])}\n")
                        else:
                            new_lines.append(f"0 {' '.join(parts[1:])}\n")
                    
                    if new_lines:
                        with open(os.path.join(dst_lbl_dir, new_txt_name), 'w') as file:
                            file.writelines(new_lines)

    # Create YAML config
    print("\nStep 4: Creating data.yaml...")
    yaml_content = f"""
path: {os.path.abspath('mega_dataset')}
train: train/images
val: valid/images

nc: 1
names: ['cardboard_box']
"""
    with open("mega_dataset/data.yaml", "w") as f:
        f.write(yaml_content)

def train_model(device_id):
    print("\nStep 5: Starting Training (This may take a while)...")
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=os.path.abspath("mega_dataset/data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="yolov8n_mega_box",
        device=device_id
    )
    
    print("\nTRAINING COMPLETE!")
    print(f"Your model is saved at: {results.save_dir}/weights/best.pt")
    print("Copy that 'best.pt' file back to your laptop!")

if __name__ == "__main__":
    device_id = check_gpu()
    setup_directories()
    datasets = download_data()
    merge_datasets(datasets)
    train_model(device_id)
