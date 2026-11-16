from pathlib import Path
import cv2 as cv

from pathlib import Path
import cv2 as cv

path = Path("License-Plate-Recognition-13", "plate_crops")

# Get all crop image paths and sort them
images = sorted(path.glob("*.jpg"))
# Select 10 images, starting from index 200
selected_images = images[30]
display = cv.imread(str(selected_images))

original_path=Path("License-Plate-Recognition-13", "train", "images")
image_paths = sorted(original_path.glob("*.jpg"))
one_image=image_paths[509]
image_read=cv.imread(str(one_image))


def bilateral_filter(image):
    filtered_image = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered_image


bilaterl=bilateral_filter(image_read)
cv.imshow('original', image_read)
cv.imshow("Bilateral Filtered Image", bilaterl)
cv.waitKey(0)
cv.destroyAllWindows()


"""
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

"""
for img in images:
    image=cv.imread(str(img))
    blur_value=calculate_blur(image)
    print(f"{img.name}: {blur_value}")

"""



