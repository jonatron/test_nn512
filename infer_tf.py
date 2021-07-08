import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img, data_format="channels_last")
model = ResNet50(weights='imagenet')
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

for i in range(3):
    t1 = time.time()
    preds = model.predict(x)
    t2 = time.time()
    print(t2 - t1)

batch_2 = np.array([x[0], x[0]])

print("batch_2")

for i in range(3):
    t1 = time.time()
    preds = model.predict(batch_2)
    t2 = time.time()
    print(t2 - t1)

print("batch_4")
batch_4 = np.array([x[0], x[0], x[0], x[0]])

for i in range(3):
    t1 = time.time()
    preds = model.predict(batch_4)
    t2 = time.time()
    print(t2 - t1)

