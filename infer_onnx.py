import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import onnxruntime as rt


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img, data_format="channels_last")
model = ResNet50(weights='imagenet')
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

providers = ['CPUExecutionProvider']
m = rt.InferenceSession("onnx_resnet.onnx", providers=providers)
output_names = ['predictions']

print("batch_1")

for i in range(5):
    t1 = time.time()
    onnx_pred = m.run(output_names, {"input": x})
    t2 = time.time()
    print('ONNX time taken: ', t2 - t1)

batch_2 = np.tile(x, (2, 1, 1, 1))

print("batch_2")

for i in range(5):
    t1 = time.time()
    onnx_pred = m.run(output_names, {"input": batch_2})
    t2 = time.time()
    print('ONNX time taken: ', t2 - t1)

print("batch_4")
batch_4 = np.tile(x, (4, 1, 1, 1))

for i in range(5):
    t1 = time.time()
    onnx_pred = m.run(output_names, {"input": batch_4})
    t2 = time.time()
    print('ONNX time taken: ', t2 - t1)

print("batch_64")
batch_64 = np.tile(x, (64, 1, 1, 1))

for i in range(3):
    t1 = time.time()
    onnx_pred = m.run(output_names, {"input": batch_64})
    t2 = time.time()
    print('ONNX time taken: ', t2 - t1)
