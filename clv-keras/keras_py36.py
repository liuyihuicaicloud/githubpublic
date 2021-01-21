# StandAlone strategy
# tensorflow 1.12.0

import os
import argparse
import tensorflow as tf
import tensorflow.keras as keras

tf.__version__

parser = argparse.ArgumentParser(description='Process args for evaluation')
parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--steps_one_epoch", type=int, default=10, help="Steps in each epochs as in training")
parser.add_argument("--prefetch", type=int, default=32, help="Size of prefetch dataset.")
parser.add_argument("--parallel", type=int, default=4, help="Numbers of parallel to fetch data.")
parser.add_argument("--num_epochs", type=int, default=30, help="The number of epochs as in training.")
parser.add_argument("--dataset_path", type=str, default='./dataset/train_v1.tfrecord',
                    help="The dataset_path  as in training.")

parser.add_argument("--model_path", type=str, default='./models',
                    help="Path to save model.")

parser.add_argument("--logs_path", type=str, default='./logs',
                    help="Path to save events.")

FLAGS = parser.parse_args()

def _get_dataset():
    dataset = tf.data.TFRecordDataset(FLAGS.dataset_path)
    dataset = dataset.map(_tf_parser,num_parallel_calls = FLAGS.parallel)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.repeat(FLAGS.num_epochs)
    dataset = dataset.prefetch(FLAGS.prefetch)
    iterator = dataset.make_one_shot_iterator()
    # create tf representation of the iterator
    image, label = iterator.get_next()
    # Bring picture back in shape
    image = tf.reshape(image, [-1,299,299,3])
    # Create a one hot array for labels
    label = tf.one_hot(label, 5)
    return image, label

def _tf_parser(record):
    features = tf.parse_single_example(record,features={
    'image_raw':tf.FixedLenFeature([],tf.string),
    'label': tf.FixedLenFeature([],tf.int64),
    'height': tf.FixedLenFeature([],tf.int64),
    'width': tf.FixedLenFeature([],tf.int64),
    'channel': tf.FixedLenFeature([],tf.int64),
    })

    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    channel = tf.cast(features['channel'],tf.int32)
    image = tf.reshape(image,[height,width,channel])
    image = tf.image.resize_images(image,[299,299],method=0)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.multiply(image, 1/255.,)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label

def main(argv=None):
    tf.reset_default_graph()
    image, label = _get_dataset()

    model_input = keras.layers.Input(tensor=image)

    model_output = keras.layers.Flatten(input_shape=(-1, 299, 299, 3))(model_input)

    model_output = keras.layers.Dense(5, activation='relu')(model_output)

    model = keras.models.Model(inputs=model_input, outputs=model_output)

    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'],target_tensors=[label])


    # callback
    t_callback = keras.callbacks.TensorBoard(log_dir='./logs')    # fit model
    model.fit(epochs=FLAGS.num_epochs,steps_per_epoch=FLAGS.steps_one_epoch,callbacks=[t_callback])

    # save to h5
    # h5file = os.path.join(FLAGS.model_path,'model.h5')

    # keras.models.save_model(model,h5file)

    # save to pb
    # tf.contrib.saved_model.save_keras_model(model,FLAGS.model_path)
    # dirs=os.listdir(FLAGS.model_path)

    # dir_name=[d for d in dirs if not d.startswith(".")]

    # dir_name=dir_name[0]

    # cmd = ("cp -r {s}/* {d}").format(s=os.path.join(FLAGS.model_path,dir_name),d=FLAGS.model_path)
    # os.system(cmd)
    # cmd = ("rm -rf {t}").format(t=os.path.join(FLAGS.model_path,dir_name))
    # os.system(cmd)
    print("Finished.")

main()
