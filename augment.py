from pathlib import Path
import cv2

images_dir = Path("License-Plate-Recognition-13/train/images")
labels_dir = Path("License-Plate-Recognition-13/train/labels")

def blur_score(image_path):
    image=cv2.imread(image_path)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian=cv2.Laplacian(gray, cv2.CV_64F)
    score=laplacian.var()
    return score
