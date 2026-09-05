# License Plate Detection with YOLOv8

Custom license plate object detection using a pretrained YOLOv8 model.

## What I Did

* Converted **COCO annotations to YOLO format**.
* Matched annotations to images using `image_id`.
* Normalized bounding-box coordinates and generated YOLO `.txt` labels.
* Organized the dataset into YOLO's `train/images`, `train/labels`, `valid/images`, and `valid/labels` structure.
* Fine-tuned **YOLOv8n** on a single license-plate class using Google Colab GPU.
* Tested the trained model with image inference.


## Dataset Format

YOLO labels use:

```text
class_id x_center y_center width height
```

with coordinates normalized between `0` and `1`.

## Tools

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · PIL · Matplotlib · Google Colab · Git/GitHub

## Workflow

**COCO dataset → annotation conversion → YOLO dataset → model fine-tuning → inference**



## tried to check if blurry training would be significantly affected by blurry images
 used the laplacian variance method but I couldn't get any appropriate range for blurry and non-blurry to use as threshold