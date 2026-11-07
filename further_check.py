from pathlib import Path
import cv2 as cv

path=Path("License-Plate-Recognition-13", "plate_crops")
images=list(path.glob("*.jpg"))
first=images[1]
display=cv.imread(str(first))



def calculate_blur(image):
    gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian=cv.Laplacian(gray, cv.CV_64F).var()
    return laplacian
calculate=calculate_blur(display)

while True:
    cv.imshow(f"{calculate}", display)
    if cv.waitKey(0) == 27:
        break
cv.destroyAllWindows()
"""
for img in images:
    image=cv.imread(str(img))
    blur_value=calculate_blur(image)
    print(f"{img.name}: {blur_value}")

"""



