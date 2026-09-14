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
    
      
    plt.figure(figsize=(8, 8))
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.colorbar(label='Signal Intensity')
    plt.title('Oil Spill Detection Pipeline - Raw Radar Band 1')
    
   
    plt.savefig('output_visualization.png', dpi=300, bbox_inches='tight')
    print("Success! Visualization saved as 'output_visualization.png'")

