from pathlib import Path
import cv2
import matplotlib.pyplot as plt

images_dir = Path("License-Plate-Recognition-13/train/images")

images=list(images_dir.iterdir())
#print(images_dir)

image_index=images[:50]
blur_scores=[]
for img in image_index:
    image = cv2.imread(str(img))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian=cv2.Laplacian(gray, cv2.CV_64F)
    score=laplacian.var()
    blur_scores.append((img, score))

blur_scores.sort(key=lambda x: x[1])
for image_path, score in blur_scores[31:41]:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(f"Score: {score:.2f}")
    plt.axis("off")
    plt.show()