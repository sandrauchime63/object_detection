from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from check import data

images_dir = Path("License-Plate-Recognition-13/train/images")

images=list(images_dir.iterdir())


def laplacian_variance(img):
    image = cv2.imread(str(img))
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian=cv2.Laplacian(gray, cv2.CV_64F)
    score=laplacian.var()
    return score








def fft_score(img):
    image = cv2.imread(str(img))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    height, width = gray.shape
    cy = height // 2
    cx = width // 2
    radius = min(cx, cy) // 4
    y, x = np.ogrid[:height, :width]
    mask = (x - cx)**2 + (y - cy)**2 > radius**2
    score = magnitude[mask].mean()

    return score


def get_plate_crops(data, images_dir):
    image_dict = {}
    for image in data['images']:
        image_dict[image['id']] = image

    plate_crops = []
    for annotation in data['annotations']:
        image_info = image_dict[annotation['image_id']]
        image_path = images_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        x, y, w, h = annotation['bbox']
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        plate = image[y:y+h, x:x+w]
        plate_crops.append((image_path, plate))

    return plate_crops
plate_crops = get_plate_crops(data, images_dir)


def tenengrad_score(img):
    image = cv2.imread(str(img))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    score = gradient.mean()

    return score

image_index=images[701]
check=tenengrad_score(plate_crops)
image = cv2.imread(str(image_index))
brg2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(brg2rgb)
plt.title(f'tenegrad score: {check}')
plt.axis('off')
plt.show()

"""

blur_lapliance=[]
tenegrad_blur=[]
fft_blur=[]
for img in images:
    lapliance=laplacian_variance(img)
    tenengrad=tenengrad_score(img)
    fft=fft_score(img)

    fft_blur.append(fft)
    blur_lapliance.append(lapliance)
    tenegrad_blur.append(tenengrad)"""


