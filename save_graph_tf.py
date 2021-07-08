from collections import OrderedDict
import struct
import pdb

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.util import nest
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.layers import Conv2D, Dense, Add, Activation, GlobalAveragePooling2D, MaxPooling2D, InputLayer

tf_rn = ResNet50(weights='imagenet')

things = {}

lines = [
    "Config Prefix=ResNet50 Platform=AVX512Float32 L1DataCachePerThread=32KiB L2CachePerThreadExL1=960KiB L3CachePerThreadExL1L2=1408KiB",
]


def input_layer(t1, inbound_layers):
    # skip padding layer
    ToTensor = 'image'
    Height = t1.input_shape[0][1]
    Width = t1.input_shape[0][2]
    Channels = t1.input_shape[0][3]
    lines.append(f"Input ToTensor={ToTensor} Channels={Channels} Height={Height} Width={Width}")


def conv(t1, inbound_layers):
    nchw_order = (3, 2, 0, 1)

    FromTensor = inbound_layers[0].name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")
    if FromTensor == "conv1Upad":
        FromTensor = "image"

    ToChannels = t1.filters

    FilterH = t1.kernel_size[0]
    FilterW = t1.kernel_size[1]
    StrideH = t1.strides[0]
    StrideW = t1.strides[1]
    # TODO: handle padding properly
    if FilterH == 7:
        PaddingH = 3
        PaddingW = 3
    elif FilterH == 3:
        PaddingH = 1
        PaddingW = 1
    else:
        PaddingH = 0
        PaddingW = 0
    DilationH = t1.dilation_rate[0]
    DilationW = t1.dilation_rate[0]
    Groups = t1.groups
    lines.append(
        f"""Conv FromTensor={FromTensor} ToTensor={ToTensor} ToChannels={ToChannels} FilterH={FilterH} FilterW={FilterW}"""
        f""" StrideH={StrideH} StrideW={StrideW} PaddingH={PaddingH} PaddingW={PaddingW} DilationH={DilationH} DilationW={DilationW} Groups={Groups}""")

    things[f"{ToTensor}Biases"] = t1.weights[1].numpy()
    things[f"{ToTensor}Weights"] = np.transpose(t1.weights[0], nchw_order)


def bn(t1, inbound_layers):
    global add_count

    FromTensor = t1.input._keras_history.layer.name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")

    lines.append(f"BatchNorm FromTensor={FromTensor} ToTensor={ToTensor} Epsilon=0.00001")

    things[f"{ToTensor}Scales"] = t1.weights[0].numpy()
    things[f"{ToTensor}Shifts"] = t1.weights[1].numpy()
    things[f"{ToTensor}Means"] = t1.weights[2].numpy()
    things[f"{ToTensor}Variances"] = t1.weights[3].numpy()


def relu(t1, inbound_layers):
    FromTensor = inbound_layers[0].name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")
    if ToTensor == "pool1pad":
        ToTensor = "pool1relu"
    lines.append(f"Activation FromTensor={FromTensor} ToTensor={ToTensor} Kind=ReLU Param=0")


def add(t1, inbound_layers):
    FromTensor = inbound_layers[0].name.replace("_", "U")
    FromTensor2 = inbound_layers[1].name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")
    lines.append(f"Add FromTensor1={FromTensor} FromTensor2={FromTensor2} ToTensor={ToTensor}")


def fc(t1, inbound_layers):
    FromTensor = inbound_layers[0].name.replace("_", "U")
    ToChannels = t1.weights[1].shape[0]
    lines.append(f"FullyConnected FromTensor={FromTensor} ToTensor=fc ToChannels={ToChannels}")
    # TODO: it could be any activation function
    lines.append("Softmax FromTensor=fc ToTensor=prob")
    lines.append("Output FromTensor=prob")

    things["fcBiases"] = t1.weights[1].numpy()
    things["fcWeights"] = np.transpose(t1.weights[0].numpy(), (1, 0))


def global_pool(t1, inbound_layers):
    FromTensor = inbound_layers[0].name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")

    # TODO: handle other types of pooling
    lines.append(f"Pooling FromTensor={FromTensor} ToTensor={ToTensor} Kind=AvgGlobal PaddingH=0 PaddingW=0")


def max_pool(t1, inbound_layers):
    FromTensor = inbound_layers[0].name.replace("_", "U")
    ToTensor = t1.name.replace("_", "U")

    if FromTensor.endswith("pad"):
        if FromTensor == "pool1Upad":
            FromTensor = "conv1Urelu"
        else:
            pdb.set_trace()

    Kind = f"Max{t1.pool_size[0]}x{t1.pool_size[1]}Stride{t1.strides[0]}"
    PaddingH = "1"  # TODO: handle padding
    PaddingW = "1"
    lines.append(f"Pooling FromTensor={FromTensor} ToTensor={ToTensor} Kind={Kind} PaddingH={PaddingH} PaddingW={PaddingW}")


# https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/utils/vis_utils.py#L234
for layer in tf_rn.layers:
    layer_id = str(id(layer))
    for i, node in enumerate(layer._inbound_nodes):
        node_key = layer.name + '_ib-' + str(i)
        if node_key in tf_rn._network_nodes:
            print("layer", layer.name)
            inbound_layers = nest.flatten(node.inbound_layers)

            if isinstance(layer, InputLayer):
                input_layer(layer, inbound_layers)
            elif isinstance(layer, Conv2D):
                conv(layer, inbound_layers)
            elif isinstance(layer, BatchNormalization):
                bn(layer, inbound_layers)
            elif isinstance(layer, Activation):
                # TODO: handle more than ReLU
                relu(layer, inbound_layers)
            elif isinstance(layer, Add):
                add(layer, inbound_layers)
            elif isinstance(layer, GlobalAveragePooling2D):
                global_pool(layer, inbound_layers)
            elif isinstance(layer, MaxPooling2D):
                max_pool(layer, inbound_layers)
            elif isinstance(layer, Dense):
                fc(layer, inbound_layers)
            else:
                print("layer not handled", layer.name, type(layer))

print("len(things.keys())", len(things.keys()))


ordered_floats = OrderedDict(sorted(things.items()))

print("ordered_floats success")

with open('float_tf_keras_3.dat', 'wb') as f:
    for key, floats in ordered_floats.items():
        floats_np = floats.flatten()
        print(key, floats.shape)
        try:
            s = struct.pack('f' * len(floats_np), *floats_np)
        except Exception as e:
            print(key)
            print(e)
            exit()
        f.write(s)


with open('resnet_tf_3.graph', 'w') as f:
    for line in lines:
        f.write(line)
        f.write("\n")


print("Saved float.dat, done.")
