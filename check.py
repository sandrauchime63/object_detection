from pathlib import Path
import shutil
import json
import torch
#from ultralytics import YOLO

path=Path("License-Plate-Recognition-13", "train")
loop=list(path.iterdir())
images=path / "images"
images.mkdir(exist_ok=True)

for file in loop:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        shutil.move(file, images / file.name)
images_dir=list(images.iterdir())
#print(images_dir[:10])

annotations_dir=path/"annotations.json"

with open(annotations_dir, 'r') as f:
    data=json.load(f)




def to_yolo_box(bbox, width, height):
   
    #width, height=images_data
    xmin, ymin, xmax, ymax=bbox
    xcentre=(xmin+ymax)/2/width
    ycentre=(ymin+ymax)/2/height
    bb_width=(xmax-xmin)/width
    bb_height=(ymax-ymin)/height
    return [xcentre, ycentre, bb_width, bb_height]

images_data=[]
#labels=[]
class_id=[]

objects=[]
def get_info(data):
    for obj in data['images']:
        #image_name=obj['file_name']
        #labels.append(image_name)
        width=obj['width']
        height=obj['height']
        class_id.append(obj['license'])
        bbox=[obj['bbox'] for obj in data['annotations']]
        yolo_box=to_yolo_box(bbox, width, height)
        objects.append(class_id + yolo_box)
    return objects
#print(get_info(data)[10])


#######bbox=[obj['bbox'] for obj in data['annotations']]
#print(data.keys())

check=[]
for i in data:
    check.append(i['annotations'][10])
print(check)

