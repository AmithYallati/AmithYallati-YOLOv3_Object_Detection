import tensorflow as tf
import numpy as np

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

def load_data():
    img_height = 416
    img_width = 416
    num_channels = 3
    num_classes = 80
    num_samples = 2

    images = np.random.rand(num_samples, img_height, img_width, num_channels).astype(np.float32)
    downsample_factor = 32
    grid_size = img_height // downsample_factor

    num_anchors = 9
    labels = np.random.rand(num_samples, grid_size, grid_size, num_anchors * (num_classes + 5)).astype(np.float32)
    return images, labels, grid_size

def custom_loss(y_true, y_pred):
    tf.print("y_true shape:", tf.shape(y_true))
    tf.print("y_pred shape:", tf.shape(y_pred))
    
    grid_size = tf.shape(y_true)[1]
    y_pred_resized = tf.image.resize(y_pred, [grid_size, grid_size])

    tf.print("Resized y_pred shape:", tf.shape(y_pred_resized))
    
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred_resized)

def main():
    images, labels, grid_size = load_data()
    model = build_yolo_model()
    model.summary()

    model.compile(optimizer='adam', loss=custom_loss)

    model.fit(
        images,
        labels,
        batch_size=2,
        epochs=10,
        validation_split=0.2
    )
    
    # Save the model
    model.save('D:/yolov3_object_detection/scripts/yolo_model.h5')  # Save model as .h5 file

if __name__ == "__main__":
    main()
