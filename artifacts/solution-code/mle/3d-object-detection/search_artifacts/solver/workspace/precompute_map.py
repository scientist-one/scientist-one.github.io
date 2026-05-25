import numpy as np
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None
print("Loading map into memory...")
map_img = np.array(Image.open('/data/train_maps/map_raster_palo_alto.png'))
print(f"Shape: {map_img.shape}, Type: {map_img.dtype}")
print("Saving as numpy array to /workspace/lyft_data/maps/map_raster_palo_alto.npy")
np.save('/workspace/lyft_data/maps/map_raster_palo_alto.npy', map_img)
print("Done!")
