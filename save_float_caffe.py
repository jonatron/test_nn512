from collections import OrderedDict
import struct
import os
import pdb

import numpy as np


import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

caffe.set_mode_cpu()

model_def = 'ResNet-50-deploy.prototxt'
# https://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel
model_weights = 'ResNet-50-model.caffemodel'

# https://github.com/BVLC/caffe/blob/master/python/draw_net.py

net_def = caffe_pb2.NetParameter()
text_format.Merge(open(model_def).read(), net_def)

layer_def_dict = {}
for layer in net_def.layer:
    layer_def_dict[layer.name] = layer

bn_count = 0
one_count = 0
one_ds_count = 0
three_count = 0
things = {}


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


print(net)


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/usr/lib/python3/dist-packages/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

'''
net.blobs['data'].reshape(5,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 224x224


image = caffe.io.load_image('elephant.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print('predicted class is:', output_prob.argmax())

# load ImageNet labels
labels_file = 'synset_words.txt'
if not os.path.exists(labels_file):
    exit("no labels")

labels = np.loadtxt(labels_file, str, delimiter='\t')

print('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print('probabilities and labels:')
print(
    list(
        zip(output_prob[top_inds], labels[top_inds])
    )
)
'''

# The activations are exposed as an OrderedDict, net.blobs.

# for each layer, show the output shape
# print("blobs")
# for layer_name, blob in net.blobs.items():
#     print(layer_name + '\t' + str(blob.data.shape))

# print('params')
# for layer_name, blobs in net.params.items():
#     for blob in blobs:
#         print(layer_name + '\t' + str(blob.data.shape))


"""
        Doc: "Generate code to apply batch normalization with per-channel mean, variance, scale, and shift parameters. " +
            "Let X be an element of FromTensor and let Y be the corresponding element of ToTensor that will be computed. " +
            "X and Y are at the same CHW coordinate in their respective tensors and the channel part of that coordinate " +
            "selects a mean M, a variance V, a scale S, and a shift H. Then Y=S*(X-M)/SQRT(V+E)+H " +
            "where E is the constant epsilon parameter (to avoid division by zero).",
"""

"""
After each BatchNorm, we have to add a Scale layer in Caffe.
The reason is that the Caffe BatchNorm layer only subtracts the mean from the input data and divides by their variance,
while does not include the γ and β parameters that respectively scale and shift the normalized distribution 1.
Conversely, the Keras BatchNormalization layer includes and applies all of the parameters mentioned above.
Using a Scale layer with the parameter “bias_term” set to True in Caffe, provides a safe trick to reproduce the exact behavior of the Keras version. 
"""

last_bn_layer_name = None
for name, layer in net.layer_dict.items():
    if layer.type == 'BatchNorm':
        # print("layer", name, layer.type)
        last_bn_layer_name = name
        bn_count += 1
        things[f"bn{bn_count}Means"] = layer.blobs[0].data
        things[f"bn{bn_count}Variances"] = layer.blobs[1].data
        print(f"bn{bn_count}Means", things[f"bn{bn_count}Means"].shape)

    elif layer.type == 'Scale':
        things[f"bn{bn_count}Scales"] = layer.blobs[0].data
        things[f"bn{bn_count}Shifts"] = layer.blobs[1].data
        print(f"bn{bn_count}Scales", things[f"bn{bn_count}Scales"].shape)

        # https://github.com/BVLC/caffe/blob/master/src/caffe/layers/scale_layer.cpp#L64
    elif layer.type == 'Convolution':
        kernel_size = (layer.blobs[0].data.shape[2], layer.blobs[0].data.shape[3])
        stride = layer_def_dict[name].convolution_param.stride
        if kernel_size == (1, 1):
            if stride == [1]:
                one_count += 1
                print(f"one{one_count}Weights", layer.blobs[0].data.shape)
                things[f"one{one_count}Weights"] = layer.blobs[0].data

                try:
                    things[f"one{one_count}Biases"] = layer.blobs[1].data
                except IndexError:
                    things[f"one{one_count}Biases"] = np.zeros(layer.blobs[0].data.shape[0])
                print(f"one{one_count}Biases", things[f"one{one_count}Biases"].shape)
                if one_count == 10:
                    assert things[f"one{one_count}Biases"].shape == (512,)
            elif stride == [2]:
                one_ds_count += 1
                things[f"oneDS{one_ds_count}Weights"] = layer.blobs[0].data
                print(f"oneDS{one_ds_count}Weights", things[f"oneDS{one_ds_count}Weights"].shape)
                try:
                    things[f"oneDS{one_ds_count}Biases"] = layer.blobs[1].data
                except IndexError:
                    things[f"oneDS{one_ds_count}Biases"] = np.zeros(layer.blobs[0].data.shape[0])
                print(f"oneDS{one_ds_count}Biases", things[f"oneDS{one_ds_count}Biases"].shape)
        elif kernel_size == (3, 3):
            three_count += 1
            things[f"three{three_count}Weights"] = layer.blobs[0].data
            print(f"three{three_count}Weights", things[f"three{three_count}Weights"].shape)
            try:
                things[f"three{three_count}Biases"] = layer.blobs[1].data
            except IndexError:
                things[f"three{three_count}Biases"] = np.zeros(layer.blobs[0].data.shape[0])
            print(f"three{three_count}Biases", things[f"three{three_count}Biases"].shape)
        elif kernel_size == (7, 7):
            things["sevenDSWeights"] = layer.blobs[0].data
            things["sevenDSBiases"] = layer.blobs[1].data
            print("sevenDSWeights", things["sevenDSWeights"].shape)
            print("sevenDSBiases", things["sevenDSBiases"].shape)
    elif layer.type == 'InnerProduct':
        things["fcBiases"] = layer.blobs[1].data
        things["fcWeights"] = layer.blobs[0].data
        print("fcBiases", things["fcBiases"].shape)
        print("fcWeights", things["fcWeights"].shape)


expected_keys = """bn10Means
bn10Scales
bn10Shifts
bn10Variances
bn11Means
bn11Scales
bn11Shifts
bn11Variances
bn12Means
bn12Scales
bn12Shifts
bn12Variances
bn13Means
bn13Scales
bn13Shifts
bn13Variances
bn14Means
bn14Scales
bn14Shifts
bn14Variances
bn15Means
bn15Scales
bn15Shifts
bn15Variances
bn16Means
bn16Scales
bn16Shifts
bn16Variances
bn17Means
bn17Scales
bn17Shifts
bn17Variances
bn18Means
bn18Scales
bn18Shifts
bn18Variances
bn19Means
bn19Scales
bn19Shifts
bn19Variances
bn1Means
bn1Scales
bn1Shifts
bn1Variances
bn20Means
bn20Scales
bn20Shifts
bn20Variances
bn21Means
bn21Scales
bn21Shifts
bn21Variances
bn22Means
bn22Scales
bn22Shifts
bn22Variances
bn23Means
bn23Scales
bn23Shifts
bn23Variances
bn24Means
bn24Scales
bn24Shifts
bn24Variances
bn25Means
bn25Scales
bn25Shifts
bn25Variances
bn26Means
bn26Scales
bn26Shifts
bn26Variances
bn27Means
bn27Scales
bn27Shifts
bn27Variances
bn28Means
bn28Scales
bn28Shifts
bn28Variances
bn29Means
bn29Scales
bn29Shifts
bn29Variances
bn2Means
bn2Scales
bn2Shifts
bn2Variances
bn30Means
bn30Scales
bn30Shifts
bn30Variances
bn31Means
bn31Scales
bn31Shifts
bn31Variances
bn32Means
bn32Scales
bn32Shifts
bn32Variances
bn33Means
bn33Scales
bn33Shifts
bn33Variances
bn34Means
bn34Scales
bn34Shifts
bn34Variances
bn35Means
bn35Scales
bn35Shifts
bn35Variances
bn36Means
bn36Scales
bn36Shifts
bn36Variances
bn37Means
bn37Scales
bn37Shifts
bn37Variances
bn38Means
bn38Scales
bn38Shifts
bn38Variances
bn39Means
bn39Scales
bn39Shifts
bn39Variances
bn3Means
bn3Scales
bn3Shifts
bn3Variances
bn40Means
bn40Scales
bn40Shifts
bn40Variances
bn41Means
bn41Scales
bn41Shifts
bn41Variances
bn42Means
bn42Scales
bn42Shifts
bn42Variances
bn43Means
bn43Scales
bn43Shifts
bn43Variances
bn44Means
bn44Scales
bn44Shifts
bn44Variances
bn45Means
bn45Scales
bn45Shifts
bn45Variances
bn46Means
bn46Scales
bn46Shifts
bn46Variances
bn47Means
bn47Scales
bn47Shifts
bn47Variances
bn48Means
bn48Scales
bn48Shifts
bn48Variances
bn49Means
bn49Scales
bn49Shifts
bn49Variances
bn4Means
bn4Scales
bn4Shifts
bn4Variances
bn50Means
bn50Scales
bn50Shifts
bn50Variances
bn51Means
bn51Scales
bn51Shifts
bn51Variances
bn52Means
bn52Scales
bn52Shifts
bn52Variances
bn53Means
bn53Scales
bn53Shifts
bn53Variances
bn5Means
bn5Scales
bn5Shifts
bn5Variances
bn6Means
bn6Scales
bn6Shifts
bn6Variances
bn7Means
bn7Scales
bn7Shifts
bn7Variances
bn8Means
bn8Scales
bn8Shifts
bn8Variances
bn9Means
bn9Scales
bn9Shifts
bn9Variances
fcBiases
fcWeights
one10Biases
one10Weights
one11Biases
one11Weights
one12Biases
one12Weights
one13Biases
one13Weights
one14Biases
one14Weights
one15Biases
one15Weights
one16Biases
one16Weights
one17Biases
one17Weights
one18Biases
one18Weights
one19Biases
one19Weights
one1Biases
one1Weights
one20Biases
one20Weights
one21Biases
one21Weights
one22Biases
one22Weights
one23Biases
one23Weights
one24Biases
one24Weights
one25Biases
one25Weights
one26Biases
one26Weights
one27Biases
one27Weights
one28Biases
one28Weights
one29Biases
one29Weights
one2Biases
one2Weights
one30Biases
one30Weights
one3Biases
one3Weights
one4Biases
one4Weights
one5Biases
one5Weights
one6Biases
one6Weights
one7Biases
one7Weights
one8Biases
one8Weights
one9Biases
one9Weights
oneDS1Biases
oneDS1Weights
oneDS2Biases
oneDS2Weights
oneDS3Biases
oneDS3Weights
oneDS4Biases
oneDS4Weights
oneDS5Biases
oneDS5Weights
oneDS6Biases
oneDS6Weights
sevenDSBiases
sevenDSWeights
three10Biases
three10Weights
three11Biases
three11Weights
three12Biases
three12Weights
three13Biases
three13Weights
three14Biases
three14Weights
three15Biases
three15Weights
three16Biases
three16Weights
three1Biases
three1Weights
three2Biases
three2Weights
three3Biases
three3Weights
three4Biases
three4Weights
three5Biases
three5Weights
three6Biases
three6Weights
three7Biases
three7Weights
three8Biases
three8Weights
three9Biases
three9Weights""".splitlines()

missing = set(expected_keys) - set(things.keys())
print("missing", missing)

assert len(things.keys()) == 320

ordered_floats = OrderedDict(sorted(things.items()))
assert list(ordered_floats.keys()) == expected_keys

print("ordered_floats success")


with open('float.dat', 'wb') as f:
    for key, floats in ordered_floats.items():
        # print(key, type(floats))
        print(key, floats.shape)
        floats_np = floats.flatten()
        try:
            s = struct.pack('f' * len(floats_np), *floats_np)
        except Exception as e:
            print(key)
            print(e)
            exit()
        f.write(s)

print("first floats:", ordered_floats['bn10Means'][0:2])

print("Saved float.dat, done.")
