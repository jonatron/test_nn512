import pdb
import struct
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img, data_format="channels_first")
print(x.shape)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(x[0].shape)

with open('elephant.dat', 'wb') as f:

    floats_np = x[0].flatten()
    print(floats_np.shape)
    try:
        s = struct.pack('f' * len(floats_np), *floats_np)
    except Exception as e:
        print(e)
        exit()
    f.write(s)


# // The inference function reads floats from (one or more) input tensors
# // and writes floats to (one or more) output tensors. All the input and
# // output tensors are owned (allocated and freed) by the caller and are
# // in CHW format, 32-bit floating point, fully packed (in other words,
# // C has the largest pitch, W has the smallest pitch, and there is no
# // padding anywhere).

from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
x = image.img_to_array(img, data_format="channels_last")
print(x.shape)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print(preds)
print('Predicted:', decode_predictions(preds, top=3)[0])
