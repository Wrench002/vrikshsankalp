import ee
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Authenticate and Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Define an Area of Interest (AOI)
aoi = ee.Geometry.Rectangle([75.5, 19.0, 77.5, 21.0])  # Example coordinates

# Load Sentinel-2 Image Collection
collection = ee.ImageCollection("COPERNICUS/S2")\
    .filterBounds(aoi)\
    .filterDate("2023-01-01", "2023-12-31")\
    .sort("CLOUDY_PIXEL_PERCENTAGE")\
    .first()

# Select relevant bands
bands = ['B4', 'B8']  # Red and NIR
image = collection.select(bands).clip(aoi)

# Compute NDVI
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Extract NDVI values
ndvi_array = ndvi.sampleRectangle(region=aoi).getInfo()['properties']['NDVI']

# Define Threshold for Tree Identification
tree_threshold = 0.4

# Tree Counting
binary_trees = np.array(ndvi_array) > tree_threshold
num_trees = np.sum(binary_trees)
print(f"Estimated Tree Count: {num_trees}")

# Vegetation Health Classification (Dummy Data)
num_samples = 1000
X_data = np.random.rand(num_samples, 64, 64, 1)  # NDVI patches
Y_data = np.random.randint(0, 3, size=(num_samples,))  # 3 health classes

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3-class classification
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Save Model
model.save("vegetation_health_cnn.h5")

# CO₂ Estimation (Simplified Formula)
co2_absorption_per_tree = 22  # kg CO₂ per year (approximation)
total_co2_absorbed = num_trees * co2_absorption_per_tree
print(f"Estimated CO₂ Absorption: {total_co2_absorbed} kg/year")

# Display NDVI Map
plt.imshow(ndvi_array, cmap='RdYlGn')
plt.colorbar()
plt.title('NDVI and Tree Density')
plt.show()
