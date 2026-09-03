from pathlib import Path
import cv2
import matplotlib.pyplot as plt

images_dir = Path("License-Plate-Recognition-13/train/images")
labels_dir = Path("License-Plate-Recognition-13/train/labels")

def blur_score(image_path):
    image=cv2.imread(image_path)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian=cv2.Laplacian(gray, cv2.CV_64F)
    score=laplacian.var()
    return score


images=list(images_dir.iterdir())
#print(images_dir)
image_index=images[10]

image = cv2.imread(str(image_index))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

score=blur_score(image_rgb)
print(score)

plt.imshow(image_rgb)
plt.title(f"Laplacian variance: {score:.2f}")
plt.axis("off")
plt.show()
"""
for image in image_path:
    run_blur=blur_score(image)
    if run_blur<threshold:
        pass
"""
