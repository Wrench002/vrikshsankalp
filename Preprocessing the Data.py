import rasterio
import numpy as np

# Read NIR and Red bands
with rasterio.open("NIR_band.tif") as nir_src:
    nir = nir_src.read(1).astype('float32')

with rasterio.open("Red_band.tif") as red_src:
    red = red_src.read(1).astype('float32')

# Compute NDVI
ndvi = (nir - red) / (nir + red)

# Save NDVI output
ndvi_meta = nir_src.meta
ndvi_meta.update(dtype='float32')

with rasterio.open("NDVI.tif", "w", **ndvi_meta) as dst:
    dst.write(ndvi, 1)
