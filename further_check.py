from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

images_dir = Path("License-Plate-Recognition-13/train/images")
labels_dir = Path("License-Plate-Recognition-13/train/labels")
images_file=list(images_dir.iterdir())[100]




label_path = labels_dir / f"{images_file.stem}.txt"

image = Image.open(images_file)
width, height = image.size

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(image)

with open(label_path) as f:
    for line in f:
        class_id, x_center, y_center, box_width, box_height = map(
            float, line.split()
        )

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x = x_center - box_width / 2
        y = y_center - box_height / 2

        rectangle = patches.Rectangle(
            (x, y),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rectangle)

ax.axis("off")
plt.show()
