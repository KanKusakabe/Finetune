"""
HOW TO USE

# Basic: Specify targets in "name:id" format (multiple targets can be separated by spaces)

# Pattern 1: YOLO-World
python create_dataset.py door.mp4 --targets "door handle:0" "button:1" --model yolo --dir my_dataset

# Pattern 2: Grounding DINO
python create_dataset.py door.mp4 --targets "door handle:0" "button:1" --model dino --dir my_dataset

# Pattern 3: Custom trained model (YOLO)
python create_dataset.py door.mp4 --targets "door handle:0" "button:1" --model custom --custom_weights runs/detect/my_model/weights/best.pt

# Pattern 4: Negative sample (video with background only)
python create_dataset.py background.mp4 --negative

# Options:
# --interval 10   : Save every 10 frames (Default: 5)
# --threshold 0.5 : Set detection threshold to 0.5 (Default: 0.35)
# --dir my_dataset : Change the output directory name to my_dataset (Default: my_dataset)

names:
  0: door handle
  1: wallet
  2: upward button
  3: downward button
"""

import argparse
import cv2
import os
import sys
import torch
import yaml
import random
import numpy as np
from PIL import Image

# Import required libraries
# transformers is only required when using DINO, but imported at the top level here
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError:
    pass # Not needed if DINO is not used

from ultralytics import YOLOWorld, YOLO

def get_args():
    if len(sys.argv) == 1:
        print("[Error] Required arguments are missing.")
        print("Expected input example:")
        print("  python create_dataset.py sample_video.mp4 --targets \"door:0\" \"window:1\" --dir my_dataset")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Create YOLO dataset (DINO / YOLO-World / Custom)")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--targets", nargs='+', type=str, help="List of targets in 'name:id' format")
    parser.add_argument("--dir", type=str, default="my_dataset", help="Output directory name")
    parser.add_argument("--interval", type=int, default=5, help="Frame interval to save")
    parser.add_argument("--threshold", type=float, default=0.35, help="Detection threshold")
    parser.add_argument("--negative", action="store_true", help="Negative sample mode")
    
    # Model Selection
    parser.add_argument("--model", type=str, default="dino", choices=["dino", "yolo", "custom"], help="Model type")
    # Path to custom weights
    parser.add_argument("--custom_weights", type=str, default=None, help="Path to custom trained .pt file (required for --model custom)")
    
    args = parser.parse_args()
    if not args.negative and not args.targets:
        parser.error("--targets is required unless --negative is set.")
    if args.model == "custom" and not args.custom_weights:
        parser.error("--custom_weights is required when using --model custom")
        
    return args

def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + (w / 2)
    cy = y1 + (h / 2)
    return cx / img_width, cy / img_height, w / img_width, h / img_height

# --- Detector Classes ---

class BaseDetector:
    def __init__(self, target_map, threshold, device):
        self.target_map = target_map
        self.threshold = threshold
        self.device = device

    def predict(self, frame):
        """Returns: list of (user_class_id, cx, cy, w, h, label_name)"""
        raise NotImplementedError

class DinoDetector(BaseDetector):
    def __init__(self, target_map, threshold, device):
        super().__init__(target_map, threshold, device)
        model_id = "IDEA-Research/grounding-dino-base"
        print(f"Loading Grounding DINO: {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.prompt_str = " . ".join(target_map.keys()) + " ."
        print(f"DINO Prompt: '{self.prompt_str}'")

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_h, img_w = frame.shape[:2]

        inputs = self.processor(images=pil_img, text=self.prompt_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.threshold,
            text_threshold=0.25, target_sizes=[pil_img.size[::-1]]
        )[0]

        detections = []
        for box, label in zip(results["boxes"].cpu().numpy(), results["text_labels"]):
            clean_label = label.strip().lower()
            if clean_label in self.target_map:
                cls_id = self.target_map[clean_label]
                cx, cy, w, h = convert_to_yolo_format(box, img_w, img_h)
                detections.append((cls_id, cx, cy, w, h, clean_label))
        return detections

class YoloWorldDetector(BaseDetector):
    def __init__(self, target_map, threshold, device):
        super().__init__(target_map, threshold, device)
        print(f"Loading YOLO-World (yolov8s-world.pt)...")
        self.model = YOLOWorld('yolov8s-world.pt') 
        self.class_names = list(target_map.keys())
        self.model.set_classes(self.class_names)

    def predict(self, frame):
        img_h, img_w = frame.shape[:2]
        results = self.model.predict(frame, conf=self.threshold, verbose=False, device=self.device)
        detections = []
        
        # Consider cases where there are no results
        if not results: return []
        
        result = results[0]
        if result.boxes is None: return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy().astype(int)

        for box, cls_idx in zip(boxes, class_indices):
            detected_name = self.class_names[cls_idx]
            if detected_name in self.target_map:
                user_id = self.target_map[detected_name]
                cx, cy, w, h = convert_to_yolo_format(box, img_w, img_h)
                detections.append((user_id, cx, cy, w, h, detected_name))
        return detections

class CustomYoloDetector(BaseDetector):
    def __init__(self, target_map, threshold, device, weights_path):
        super().__init__(target_map, threshold, device)
        print(f"Loading Custom Model: {weights_path}...")
        self.model = YOLO(weights_path)
        
        # Class names defined in the model (e.g., {0: 'door', 1: 'handle'})
        self.model_names = self.model.names 
        print(f"Model trained classes: {self.model_names}")

    def predict(self, frame):
        img_h, img_w = frame.shape[:2]
        # Perform inference with custom model
        results = self.model.predict(frame, conf=self.threshold, verbose=False, device=self.device)
        detections = []

        if not results: return []
        result = results[0]
        if result.boxes is None: return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy().astype(int)

        for box, cls_idx in zip(boxes, class_indices):
            # Retrieve the class name detected by the model (e.g., "door")
            detected_name = self.model_names[cls_idx]
            
            # Remap to the user-specified ID
            # This ensures they are saved with correct IDs if names match, even if model IDs differ
            if detected_name in self.target_map:
                user_id = self.target_map[detected_name]
                cx, cy, w, h = convert_to_yolo_format(box, img_w, img_h)
                detections.append((user_id, cx, cy, w, h, detected_name))
        
        return detections

# --- Main Logic ---

def update_yaml_multi(yaml_path, base_dir, target_map, is_negative):
    data = {
        "path": os.path.abspath(base_dir),
        "train": "images",
        "val": "images",
        "names": {}
    }

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            try:
                loaded_data = yaml.safe_load(f)
                if loaded_data:
                    data = loaded_data
                    if "names" not in data:
                        data["names"] = {}
            except yaml.YAMLError:
                pass

    if not is_negative:
        if isinstance(data["names"], list):
            data["names"] = {i: name for i, name in enumerate(data["names"])}
        for name, cid in target_map.items():
            data["names"][cid] = name

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    args = get_args()

    # Parse target information
    target_map = {}
    if not args.negative:
        print("Targets:")
        for t in args.targets:
            try:
                name, cid = t.rsplit(':', 1)
                name = name.strip().lower() # Names are managed in lowercase
                cid = int(cid)
                target_map[name] = cid
                print(f" - {name} (ID: {cid})")
            except ValueError:
                print(f"Error: Invalid format '{t}'")
                return

    # Directories
    base_dir = args.dir
    dirs = {
        "img": os.path.join(base_dir, "images"),
        "lbl": os.path.join(base_dir, "labels"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Initialize Detector
    detector = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.negative:
        if args.model == "dino":
            detector = DinoDetector(target_map, args.threshold, device)
        elif args.model == "yolo":
            detector = YoloWorldDetector(target_map, args.threshold, device)
        elif args.model == "custom":
            detector = CustomYoloDetector(target_map, args.threshold, device, args.custom_weights)

    # Video processing
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    frame_count = 0
    saved_count = 0
    video_basename = os.path.basename(args.video_path).rsplit('.', 1)[0]

    print("\nStarting processing... (Press Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % args.interval == 0:
                filename = f"{video_basename}_frame_{frame_count:06d}"
                img_save_path = os.path.join(dirs["img"], filename + ".jpg")
                lbl_save_path = os.path.join(dirs["lbl"], filename + ".txt")
                
                should_save = False
                detected_names = []
                lines = []

                if args.negative:
                    should_save = True
                else:
                    detections = detector.predict(frame)
                    if detections:
                        for (cid, cx, cy, w, h, name) in detections:
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            detected_names.append(name)
                        should_save = True

                if should_save:
                    cv2.imwrite(img_save_path, frame)
                    with open(lbl_save_path, "w") as f:
                        f.write("\n".join(lines))
                    
                    saved_count += 1
                    msg = "Negative Sample" if args.negative else f"Found: {detected_names}"
                    print(f"Saved: {filename} -> {msg}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        update_yaml_multi(os.path.join(base_dir, "data.yaml"), base_dir, target_map, args.negative)
        print(f"Done! Saved {saved_count} images.")

if __name__ == "__main__":
    main()
