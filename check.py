from pathlib import Path
import shutil
import json


# put the paths in a variable identifier
path=Path("License-Plate-Recognition-13", "train")
##path=Path("License-Plate-Recognition-13", "test")##
##path=Path("License-Plate-Recognition-13", "valid")##
loop=list(path.iterdir())
images=path / "images"
images.mkdir(exist_ok=True)

## put all the images in one folder
for file in loop:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        shutil.move(file, images / file.name)
images_dir=list(images.iterdir())

## load the annotations to convert to YOLO format
annotations_dir=path/"_annotations.coco.json"

with open(annotations_dir, 'r') as f:
    data=json.load(f)

####dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])
############print(data['annotations'][10])
#{'id': 11, 'image_id': 10, 'category_id': 1, 
#'bbox': [159, 91, 96.63, 85.711], 'area': 8282.233, 'segmentation': [], 'iscrowd': 0}
############print(data['images'][10])
#{'id': 10, 'license': 1, 
# 'file_name': 'xemay1235_jpg.rf.9ec4ba864fe9c996c3c4ff30ff7f37f3.jpg', 
# 'height': 294, 'width': 460, 'date_captured': '2026-01-26T09:42:31+00:00', 
# 'extra': {'name': 'xemay1235.jpg'}}


## prepare the bounding box data in yolo format
def yolo_bbox(bbox, width, height):
    x, y, w, h=bbox
    x_center = (x + w)/2 / width
    y_center = (y + h)/2 / height
    bb_width = w / width
    bb_height = h / height
    return [x_center, y_center, bb_width, bb_height]

##Link the annotations to their matching images 
def get_info(data):
    image_dict={}
    check={}
    for image in data['images']:
        image_dict[image['id']]=image
    for annotation in data['annotations']:
        image=image_dict[annotation['image_id']]
        name=image['file_name']
        width=image['width']
        height=image['height']
        bbox=annotation['bbox']
        id=0
        yolo_box=yolo_bbox(bbox, width, height)
        label=[id] + yolo_box
        if name not in check:
            check[name]=[]
        check[name].append(label)


    return check



check=get_info(data)





labels_dir=path/'labels'
labels_dir.mkdir(exist_ok=True)

##create the file for yolo with the xcentre, ycentre, width 
# and height
def create_file(result):
    for filename, label in result.items():
        name=labels_dir / Path(filename).with_suffix(".txt")
        with open(name, 'w') as f:
            for lab in label:
                f.write(" ".join(str(x) for x in lab))
                f.write("\n")

           
see=create_file(check)





            
            






       

            
