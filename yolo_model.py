import tensorflow as tf  # Ensure TensorFlow is imported

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(0.0005)}
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1)
    ])

def yolo_body(inputs, num_anchors, num_classes):
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)
    x = DarknetConv2D_BN_Leaky(64, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(128, (3, 3))(x)
    outputs = tf.keras.layers.Conv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    return tf.keras.Model(inputs, outputs)

def build_yolo_model(input_shape=(416, 416, 3), num_classes=80):
    inputs = tf.keras.Input(shape=input_shape)
    model_body = yolo_body(inputs, num_anchors=9, num_classes=num_classes)
    return model_body

if __name__ == "__main__":
    model = build_yolo_model()
    model.summary()  # This line should print the model summary
