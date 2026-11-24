from pathlib import Path

path=Path('Nigeria_License_Plates.v1-license_plate.yolo26', 'train')

labels=path/'labels'
for file in labels.iterdir():
    with open(file, 'r') as f:
        line=f.readline()
    print(line)
    break

normal_path=Path('License-Plate-Recognition-13', 'train')
labels=normal_path/'labels'
for file in labels.iterdir():
    with open(file, 'r') as f:
        line=f.readline()
    print(line)
    break