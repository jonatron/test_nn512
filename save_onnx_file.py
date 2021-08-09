import time

import tensorflow as tf
import tf2onnx
import onnx
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import onnxruntime as rt

# pip install onnxruntime pillow tf2onnx

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img, data_format="channels_last")
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = ResNet50(weights='imagenet')
for i in range(10):
    t1 = time.time()
    preds = model.predict(x)
    t2 = time.time()
    print('Keras Predicted:', decode_predictions(preds, top=3)[0])
    print('Keras time taken: ', t2 - t1)

input_signature = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
output_names = [n.name for n in onnx_model.graph.output]
print("output_names", output_names)
onnx.save(onnx_model, "onnx_resnet.onnx")
providers = ['CPUExecutionProvider']
m = rt.InferenceSession("onnx_resnet.onnx", providers=providers)

for i in range(10):
    t1 = time.time()
    onnx_pred = m.run(output_names, {"input": x})
    t2 = time.time()
    print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])
    print('ONNX time taken: ', t2 - t1)


