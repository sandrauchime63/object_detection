from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

images_dir = Path("License-Plate-Recognition-13/train/images")

images=list(images_dir.iterdir())


def laplacian_variance(img):
    image = cv2.imread(str(img))
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian=cv2.Laplacian(gray, cv2.CV_64F)
    score=laplacian.var()
    return score



def tenengrad_score(img):
    image = cv2.imread(str(img))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    score = gradient.mean()

    return score

image_index=images[701]
check=tenengrad_score(image_index)
image = cv2.imread(str(image_index))
brg2rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(brg2rgb)
plt.title(f'tenegrad score: {check}')
plt.axis('off')
plt.show()




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

blur_lapliance=[]
tenegrad_blur=[]
fft_blur=[]
for img in images:
    lapliance=laplacian_variance(img)
    tenengrad=tenengrad_score(img)
    fft=fft_score(img)

    fft_blur.append(fft)
    blur_lapliance.append(lapliance)
    tenegrad_blur.append(tenengrad)


