import tensorflow as tf

def preprocess_image(serialized_examples):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    examples = tf.io.parse_example(serialized_examples, feature_description)
    targets = examples.pop('label')
    images = tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'], channels=3), tf.float32), 299, 299)
    images = keras.applications.xception.preprocess_input(images)
    return images, targets