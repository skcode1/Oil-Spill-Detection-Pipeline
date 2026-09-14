import tifffile as tiff
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Use the full absolute path to your file
file_path = "/home/marcist/Projects/Oil-Spill-Detection-Pipeline-/SIH26143/data/raw/zenodo_part3/Images/No oil/00000.tif"

if not os.path.exists(file_path):
    print(f"Error: The file was not found at {file_path}")
    print("Please double-check the folder names in your path.")
else:
    # 2. Read the GeoTIFF data array
    img = tiff.imread(file_path)
    
    print("Image shape:", img.shape)  # Should show (2048, 2048, 2)
    print("Data type:", img.dtype)    # Should show float32
    
      
    masked_band = img[:, :, 0]
active_pixels = masked_band[masked_band < 0] # Filter out the 0/No-data values

# Dynamically calculate the 2nd and 98th percentiles to eliminate extreme outliers
vmin = np.percentile(active_pixels, 2)
vmax = np.percentile(active_pixels, 98)

plt.figure(figsize=(8, 8))
# Apply vmin and vmax parameters to stretch the color distribution 
plt.imshow(masked_band, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar(label='Signal Intensity (dB - Contrast Stretched)')
plt.title('Oil Spill Detection Pipeline - Contrast Enhanced')
plt.savefig('output_stretched.png', dpi=300, bbox_inches='tight')

