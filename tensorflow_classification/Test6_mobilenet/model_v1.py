from pydoc import classname
import tensorflow as tf

# 导入所有必要的层

from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense

from tensorflow.keras import Model

# MobileNet block


def mobilnet_block(x, filters, strides):

    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# 模型主干


def MobileNetV1(im_height=224,
                im_width=224,
                num_classes=1000,
                include_top=True):

    input = Input(shape=(im_height, im_width, 3))

    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 模型的主要部分

    x = mobilnet_block(x, filters=64, strides=1)
    x = mobilnet_block(x, filters=128, strides=2)
    x = mobilnet_block(x, filters=128, strides=1)
    x = mobilnet_block(x, filters=256, strides=2)
    x = mobilnet_block(x, filters=256, strides=1)
    x = mobilnet_block(x, filters=512, strides=2)

    for _ in range(5):
        x = mobilnet_block(x, filters=512, strides=1)

    x = mobilnet_block(x, filters=1024, strides=2)
    x = mobilnet_block(x, filters=1024, strides=1)
    if include_top is True:

        x = AvgPool2D(pool_size=7, strides=1, data_format='channels_first')(x)
        output = Dense(units=num_classes, activation='softmax')(x)
    else:
        output = x

    model = Model(inputs=input, outputs=output)

    return model


if __name__ == '__main__':
    model = MobileNetV1(num_classes=5)
    model.summary()

    # 绘制模型
    # sudo apt-get install graphviz
    tf.keras.utils.plot_model(model, to_file='mobilenet_v1.png', show_shapes=True,
                              show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
