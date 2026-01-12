from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

PATH_TO_RAW_CLASS_IMAGES = "dataset/raw"

class DataSplitter:

    def __init__(self, raw_img_dir_path):
        self.raw_img_dir = Path(raw_img_dir_path)
    
    def get_raw_data(self):
        X_gesture_images = []
        y_gesture_classes = []
        gesture_directories = list(self.raw_img_dir.iterdir())
        for gesture_dir in gesture_directories:
            gesture_class = gesture_dir.name
            print(f"\nNow processing -> Gesture: {gesture_class}")
            gesture_images = list(gesture_dir.glob("*.jpg"))
            for img in gesture_images:
                X_gesture_images.append(img)
                y_gesture_classes.append(gesture_class)
        
        print(f"\nAll images now in 1 list and corresponding class labels in another list ->")
        print(f"\nList containing all image paths -> X_gesture_images\nSize: {len(X_gesture_images)} image paths.")
        print(f"\nList containing all class labels -> y_gesture_classes\nSize: {len(y_gesture_classes)} labels.")

        
