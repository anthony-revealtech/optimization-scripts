import tensorflow as tf
import numpy as np

MODEL_PATH = "aliked-n16_fp16.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()

fp32_tensors = []

for t in tensor_details:
    if t["dtype"] == np.float32:
        fp32_tensors.append({
            "name": t["name"],
            "index": t["index"],
            "shape": t["shape"]
        })

if fp32_tensors:
    print("❌ FP32 tensors detected:")
    for t in fp32_tensors:
        print(f'  - Name: {t["name"]}, Index: {t["index"]}, Shape: {t["shape"]}')
    raise RuntimeError("Model contains FP32 tensors (activations or buffers). Not pure FP16.")

print("✅ Model is pure FP16 (no FP32 tensors detected).")


