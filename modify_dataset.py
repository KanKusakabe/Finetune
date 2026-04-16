import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import glob
import os
import yaml
import argparse

# ==========================================
# Configuration
# ==========================================
# DATASET_DIR = "dataset/wallet11"  # Root directory of the dataset
DATASET_DIR = "my_dataset"  # Root directory of the dataset
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720
COLOR_MAP = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "brown", "pink"]

class App:
    def __init__(self, root, dataset_dir, duplicated=1):
        self.root = root
        self.root.title("Simple YOLO Annotator")
        self.dataset_dir = dataset_dir
        self.duplicated = duplicated

        # --- Load Data ---
        # Support both "images/*.jpg" and "images/*/*.jpg"
        img_pattern1 = os.path.join(dataset_dir, "images", "*.jpg")
        img_pattern2 = os.path.join(dataset_dir, "images", "*", "*.jpg")
        all_img_files = sorted(glob.glob(img_pattern1) + glob.glob(img_pattern2))
        
        self.img_files = []
        for img_path in all_img_files:
            label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
            label_counts = {}
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                label_counts[cid] = label_counts.get(cid, 0) + 1
                except:
                    pass
            
            # Check if there are labels duplicated more than 'duplicated' times
            if self.duplicated <= 0:
                self.img_files.append(img_path)
            else:
                if any(c >= self.duplicated for c in label_counts.values()):
                    self.img_files.append(img_path)
        
        if not self.img_files:
            messagebox.showerror("Error", "No images found.")
            root.destroy()
            return
        
        self.class_names = self.load_classes()
        self.current_idx = 0
        self.current_cls_id = 0  # Class ID for the current drawing mode
        
        # --- UI Setup ---
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Event bindings
        self.root.bind("<space>", self.next_image)
        self.root.bind("n", self.next_image)
        self.root.bind("b", self.prev_image)
        self.root.bind("d", self.delete_image)
        self.root.bind("q", self.quit_app)
        self.root.bind("<Key>", self.handle_key) # For number keys
        
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click) # Right click to delete

        # Drawing state management
        self.boxes = [] # [(cls_id, x1, y1, x2, y2), ...] (pixel coordinates)
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        
        # Image display variables
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.load_image()

    def load_classes(self):
        """Get class names from data.yaml"""
        yaml_path = os.path.join(self.dataset_dir, "data.yaml")
        names = {}
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if "names" in data:
                        raw_names = data["names"]
                        if isinstance(raw_names, list):
                            names = {i: n for i, n in enumerate(raw_names)}
                        else:
                            names = raw_names
            except:
                pass
        return names

    def load_image(self):
        if self.current_idx >= len(self.img_files):
            messagebox.showinfo("Done", "All images have been checked.")
            self.root.destroy()
            return

        self.img_path = self.img_files[self.current_idx]
        self.label_path = self.img_path.replace("images", "labels").replace(".jpg", ".txt")

        # Load image & calculate resize
        pil_img = Image.open(self.img_path)
        w, h = pil_img.size
        
        # Calculate scale to fit within the window
        scale_w = WINDOW_WIDTH / w
        scale_h = WINDOW_HEIGHT / h
        self.scale = min(scale_w, scale_h, 1.0) # Do not enlarge
        
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        self.tk_img = ImageTk.PhotoImage(pil_img.resize((new_w, new_h)))
        
        # Align left (vertically centered)
        self.offset_x = 0
        self.offset_y = (WINDOW_HEIGHT - new_h) // 2
        self.orig_w, self.orig_h = w, h

        # Load labels
        self.boxes = []
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # YOLO (normalized) -> Pixel (original image)
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        self.boxes.append([cid, x1, y1, x2, y2])

        self.redraw()

    def save_labels(self):
        """Save the current boxes to a file"""
        lines = []
        for (cid, x1, y1, x2, y2) in self.boxes:
            # Pixel -> YOLO (normalized)
            cx = (x1 + x2) / 2 / self.orig_w
            cy = (y1 + y2) / 2 / self.orig_h
            bw = abs(x2 - x1) / self.orig_w
            bh = abs(y2 - y1) / self.orig_h
            
            # Clip bounds
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))
            
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
        
        if lines:
            with open(self.label_path, 'w') as f:
                f.write("\n".join(lines))
        else:
            # If there are no boxes, create an empty file or delete the file
            # Here we default to creating an empty file
            with open(self.label_path, 'w') as f:
                pass

    def redraw(self):
        self.canvas.delete("all")
        # Draw image
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_img)
        
        # Draw boxes
        for i, box in enumerate(self.boxes):
            cid, x1, y1, x2, y2 = box
            
            # Convert to display coordinates
            sx1 = x1 * self.scale + self.offset_x
            sy1 = y1 * self.scale + self.offset_y
            sx2 = x2 * self.scale + self.offset_x
            sy2 = y2 * self.scale + self.offset_y
            
            color = COLOR_MAP[cid % len(COLOR_MAP)]
            
            # Rectangle
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2, tags=f"box_{i}")
            
            # Label name
            label_name = self.class_names.get(cid, str(cid))
            self.canvas.create_text(sx1, sy1-10, text=f"{label_name} ({cid})", fill=color, anchor="w", font=("Arial", 12, "bold"))

        # Count the number of labels in the current image
        label_counts = {}
        for box in self.boxes:
            cid = box[0]
            label_counts[cid] = label_counts.get(cid, 0) + 1
            
        if label_counts:
            counts_str = " / ".join([f"{self.class_names.get(k, str(k))}: {v}" for k, v in sorted(label_counts.items())])
        else:
            counts_str = "None"

        # Display information
        info = f"[{self.current_idx+1}/{len(self.img_files)}] Space/N:Next / B:Back / D:Del Img / Q:Quit / R-Click:Del Box"
        cls_info = f"Current Draw Class: {self.class_names.get(self.current_cls_id, self.current_cls_id)} (Press 0-9 to change)"
        self.canvas.create_text(10, 20, text=info, fill="white", anchor="w", font=("Arial", 14))
        self.canvas.create_text(10, 50, text=cls_info, fill="yellow", anchor="w", font=("Arial", 14))
        self.canvas.create_text(10, 80, text=f"File: {os.path.basename(self.img_path)}", fill="white", anchor="w", font=("Arial", 14))
        self.canvas.create_text(10, 110, text=f"Labels in this image: {counts_str}", fill="cyan", anchor="w", font=("Arial", 14, "bold"))
        self.root.update_idletasks()

    # --- Operation Events ---
    def handle_key(self, event):
        if event.char.isdigit():
            self.current_cls_id = int(event.char)
            self.redraw()

    def next_image(self, event=None):
        self.save_labels()
        self.current_idx += 1
        self.load_image()

    def prev_image(self, event=None):
        self.save_labels() # Save the current state before going back
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def delete_image(self, event=None):
        # Delete files
        try:
            os.remove(self.img_path)
            if os.path.exists(self.label_path):
                os.remove(self.label_path)
            print(f"Deleted: {self.img_path}")
            # Remove from list and proceed (keep the index unchanged)
            del self.img_files[self.current_idx]
            self.load_image()
        except Exception as e:
            print(f"Error deleting: {e}")

    def quit_app(self, event=None):
        self.save_labels()
        self.root.destroy()

    # --- Mouse Operations (Create Box) ---
    def on_mouse_down(self, event):
        # Convert coordinates to image scale
        img_x = (event.x - self.offset_x) / self.scale
        img_y = (event.y - self.offset_y) / self.scale
        
        # Ignore if outside the image
        if 0 <= img_x <= self.orig_w and 0 <= img_y <= self.orig_h:
            self.start_x = img_x
            self.start_y = img_y
            
            # Show a temporary rectangle
            self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", width=2, dash=(4, 4))

    def on_mouse_drag(self, event):
        if self.rect_id:
            x1 = self.start_x * self.scale + self.offset_x
            y1 = self.start_y * self.scale + self.offset_y
            self.canvas.coords(self.rect_id, x1, y1, event.x, event.y)

    def on_mouse_up(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            
            img_x = (event.x - self.offset_x) / self.scale
            img_y = (event.y - self.offset_y) / self.scale
            
            # Add if it has a reasonable size
            if abs(img_x - self.start_x) > 5 and abs(img_y - self.start_y) > 5:
                x1, x2 = sorted([self.start_x, img_x])
                y1, y2 = sorted([self.start_y, img_y])
                
                # Clip bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(self.orig_w, x2); y2 = min(self.orig_h, y2)

                self.boxes.append([self.current_cls_id, x1, y1, x2, y2])
                self.redraw()

    # --- Mouse Operations (Delete Box) ---
    def on_right_click(self, event):
        # Find and delete the box at the clicked position
        img_x = (event.x - self.offset_x) / self.scale
        img_y = (event.y - self.offset_y) / self.scale
        
        # Check in reverse order (newly drawn items are in the front)
        for i in range(len(self.boxes) - 1, -1, -1):
            cid, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.boxes.pop(i)
                self.redraw()
                return # Delete only one item per click

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("[Error] Required arguments are missing.")
        print("Expected input example:")
        print("  python modify_dataset.py --dir my_dataset")
        print("  python modify_dataset.py --dir my_dataset --duplicated 2")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Simple YOLO Annotator")
    parser.add_argument("--dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--duplicated", type=int, default=1, help="Filter to show only images having at least one label duplicated this many times (default: 1)")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    app = App(root, args.dir, args.duplicated)
    root.mainloop()