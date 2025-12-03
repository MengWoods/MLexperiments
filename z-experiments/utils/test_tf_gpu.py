import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import time

# --- 1. Verify GPU Availability ---
print("--- 1. Checking for available GPUs ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Optionally, set memory growth to avoid pre-allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"✅ Success! TensorFlow found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
             print(f"   - GPU {i}: {gpu}")
        
    except RuntimeError as e:
        # Must be run before GPUs have been initialized
        print(f"❌ Error during GPU configuration: {e}")
else:
    print("⚠️ Warning: No GPU devices found by TensorFlow.")

# --- 2. Enable Device Logging for Verification ---
# Set the environment variable to enable all logs (0) 
# and explicitly enable device placement logging in the session.
print("\n--- 2. Enabling Device Placement Logging ---")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
tf.debugging.set_log_device_placement(True)
print("Device placement logging enabled. Look for 'device:GPU:0' in the output logs.")
print("The lines like 'Executing op ... in device /job:localhost/replica:0/task:0/device:GPU:0' mean it's running on the GPU.")


# --- 3. Define and Train a Minimal Keras Model ---
print("\n--- 3. Training a Minimal Model on Dummy Data ---")

# Define a small Sequential Model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid') 
])

# Create dummy data
# A small dataset to quickly check functionality
x_train = np.random.rand(100, 10).astype('float32')
y_train = np.random.randint(0, 2, size=(100, 1)).astype('float32')

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Start a timer
start_time = time.time()

# Train the model for a few steps
# The log messages confirming GPU usage should appear during this step.
print("\nStarting model fit...")
model.fit(
    x_train, 
    y_train, 
    epochs=5, 
    batch_size=16, 
    verbose=0 # Set to 1 for full progress bar, 0 for minimal output
)

end_time = time.time()

# --- 4. Final Output ---
print("\n--- 4. Test Summary ---")
print("✅ Training finished.")
print(f"Total training time: {end_time - start_time:.4f} seconds.")
print("If GPU(s) were found and device placement logging showed 'device:GPU:0', then your setup is likely working.")