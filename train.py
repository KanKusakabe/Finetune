"""
HOW TO USE:
$ python train.py path/to/yaml.yaml
"""

import argparse
import os
import sys
import yaml
import random
import glob
from ultralytics import YOLO

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_master_yaml(config, val_ratio):
    """
    Automatically generate an integrated YAML for YOLO from the dataset path list in config.
    Collects image lists and dynamically splits them into train and val at runtime.
    """
    dataset_dirs = config.get('datasets', [])
    if not dataset_dirs:
        print("[Error] 'datasets' is not defined in the config file.")
        sys.exit(1)

    all_images = []
    names = None
    nc = 0

    print(f"\n[Info] Integrating {len(dataset_dirs)} datasets...")

    for d_dir in dataset_dirs:
        # Look for data.yaml in each dataset
        sub_yaml_path = os.path.join(d_dir, 'data.yaml')
        
        if not os.path.exists(sub_yaml_path):
            print(f"[Error] data.yaml not found: {sub_yaml_path}")
            sys.exit(1)

        data = load_yaml(sub_yaml_path)

        # 1. Collect existing image (jpg) paths
        images_dir = os.path.join(d_dir, "images")
        if os.path.exists(images_dir):
            jpg_files = glob.glob(os.path.join(images_dir, "*.jpg"))
            all_images.extend([os.path.abspath(p) for p in jpg_files])

        # 2. Get class names (assuming the first dataset is correct)
        if names is None:
            names = data.get('names')
            nc = data.get('nc', len(names))
        else:
            # Check if class names match just in case (optional)
            if data.get('names') != names:
                print(f"[Warning] Potential class definition mismatch: {d_dir}")

    if not all_images:
        print("[Error] Target image files (*.jpg) not found.")
        sys.exit(1)

    # 3. Shuffle the list and split into train/val
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1.0 - val_ratio))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"[Info] Total images: {len(all_images)} (Train: {len(train_images)}, Val: {len(val_images)})")

    # 4. Export to text files in YOLO format
    train_txt_path = os.path.abspath('generated_train.txt')
    val_txt_path = os.path.abspath('generated_val.txt')

    with open(train_txt_path, 'w', encoding='utf-8') as f:
        # Write file paths to the list, separated by newlines
        f.write('\n'.join(train_images))
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_images))

    # 5. Integrated data dictionary (specify absolute paths for text files)
    master_data = {
        'train': train_txt_path,
        'val': val_txt_path,
        'nc': nc,
        'names': names
    }

    # Save as a temporary file
    output_path = 'generated_master_data.yaml'
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(master_data, f, allow_unicode=True)
    
    print(f"[Info] Generated YOLO configuration file: {output_path}")
    return output_path

def main():
    if len(sys.argv) == 1:
        print("[Error] Required arguments are missing.")
        print("Expected input example:")
        print("  python train.py train_config.yaml")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='YOLOv8 YAML configuration based training script')
    parser.add_argument('config', type=str, help='Path to the training configuration YAML file (e.g., train_config.yaml)')

    args = parser.parse_args()

    # 1. Load configuration file
    if not os.path.exists(args.config):
        print(f"[Error] Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_yaml(args.config)

    # 2. Get val_ratio (default is 0.2)
    val_ratio = float(config.get('val_ratio', 0.2))

    # 3. Combine multiple datasets to create a single YAML (including generating train.txt and val.txt)
    master_yaml_path = create_master_yaml(config, val_ratio)

    # 4. Read training settings (use default values if not defined in YAML)
    weights  = config.get('weights', 'YOLO26m.pt') # Default format if unspecified
    project  = config.get('project', 'my_saved_models')
    name     = config.get('name', 'my_custom_model')
    epochs   = config.get('epochs', 100)
    batch    = config.get('batch', 16)
    imgsz    = config.get('imgsz', 640)
    rect     = config.get('rect', False)
    exist_ok = config.get('exist_ok', False)

    print(f"\n=== Start Training: {name} ===")
    
    # 5. Load model and start training
    model = YOLO(weights)
    
    try:
        model.train(
            data=master_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            project=project,
            rect=rect,
            exist_ok=exist_ok,
            workers = 4
        )
    except Exception as e:
        print(f"\n[Error] An error occurred during training: {e}")
        print("Hint: Ensure the labels folder is placed correctly.")
    finally:
        # Keeping the generated temporary files is useful for debugging, but enable the following if you want to delete them
        # if os.path.exists(master_yaml_path):
        #     os.remove(master_yaml_path)
        pass

if __name__ == "__main__":
    main()
