import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
img = np.array(Image.open('/data/train_maps/map_raster_palo_alto.png'))
print(img.shape)
