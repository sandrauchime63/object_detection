from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET



path=Path('License-Plate-Recognition-13', 'train')
images = path / 'images'

for img in set(images.glob("*")):
	pass



