from pathlib import Path
import cv2 as cv

from pathlib import Path
import cv2 as cv

path = Path("License-Plate-Recognition-13", "plate_crops")

# Get all crop image paths and sort them
images = sorted(path.glob("*.jpg"))

# Select 10 images, starting from index 200
selected_images = images[300:310]


def calculate_blur(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F).var()
    return laplacian


def tenengrad(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv.magnitude(gx, gy)
    return gradient_magnitude.var()
for i, image_path in enumerate(selected_images):

    display = cv.imread(str(image_path))

    if display is None:
        continue

    score = tenengrad(display)

    window_name = f"Blur score: {score:.2f} - {image_path.name}"

   
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 300, 220)


    x = (i % 5) * 320
    y = (i // 5) * 260

    cv.moveWindow(window_name, x, y)
    cv.imshow(window_name, display)
    print(window_name)
cv.waitKey(0)
cv.destroyAllWindows()
"""
for img in images:
    image=cv.imread(str(img))
    blur_value=calculate_blur(image)
    print(f"{img.name}: {blur_value}")

"""



