from pathlib import Path
import json
from collections import defaultdict
import cv2

path=Path("License-Plate-Recognition-13", "train")
images_dir=path/"images"

annotation=path/ "_annotations.coco.json"
with open(annotation, 'r') as f:
    data=json.load(f)
    #data.keys=dict_keys(['info', 'licenses', 'categories', 'images', 'annotations']) 
#data["images"]["id"]=data["annotation"]["image_id"]

def save_plate_crops(data, images_dir, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    image_dict = {
        img['id']: img for img in data['images']
    }

    saved_count = 0

    for annotation in data['annotations']:

        image_info = image_dict[annotation['image_id']]
        image_path = images_dir / image_info['file_name']

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        x, y, w, h = map(int, annotation['bbox'])

        height, width = image.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        original_name = Path(image_info['file_name']).stem
        annotation_id = annotation['id']

        crop_filename = f"{original_name}_ann_{annotation_id}.jpg"

        crop_path = output_dir / crop_filename

        cv2.imwrite(str(crop_path), plate)

        saved_count += 1

    print(f"Saved {saved_count} plate crops to {output_dir}")

output_dir = Path("License-Plate-Recognition-13/plate_crops")

#save_plate_crops(data, images_dir, output_dir)

