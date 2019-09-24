"""An example of training Keras model with ps-worker strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Process args for evaluation')
parser.add_argument("--model_path", type=str, default='/output/models',
                    help="Path to save model.")

FLAGS = parser.parse_args()

def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(100)
  dataset = dataset.batch(32)
  return dataset

def serving_input_receiver_fn():
  inputs = {'dense_input': tf.placeholder(tf.float32,[None, 1024,10])}
  return tf.estimator.export.ServingInputReceiver(inputs,inputs)


def main():

  print("sleep 5s......")
  time.sleep(5)
  model_dir = FLAGS.model_path
  print('Using %s to store checkpoints.' % model_dir)

  # Define a Keras Model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  # Compile the model.
  optimizer = tf.train.GradientDescentOptimizer(0.2)
  model.compile(loss='binary_crossentropy', optimizer=optimizer)
  model.summary()
  tf.keras.backend.set_learning_phase(True)

  # Define DistributionStrategies and convert the Keras Model to an
  # Estimator that utilizes these DistributionStrateges.
  # Evaluator is a single worker, so using MirroredStrategy.
  config = tf.estimator.RunConfig(
      experimental_distribute=tf.contrib.distribute.DistributeConfig(
          train_distribute=tf.contrib.distribute.ParameterServerStrategy(
              num_gpus_per_worker=0),
          eval_distribute=tf.contrib.distribute.MirroredStrategy(
              num_gpus_per_worker=0)))
  ckpt_dir = os.path.join(model_dir,'ckpt')
  keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, config=config,model_dir=ckpt_dir)

  # Train and evaluate the model. Evaluation will be skipped if there is not an
  # "evaluator" job in the cluster.
  tf.estimator.train_and_evaluate(
      keras_estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))

  if config.is_chief:
    print("chief worker, save model")
    try:
      keras_estimator.export_savedmodel(model_dir, serving_input_receiver_fn)
      cmd = ("rm -rf {t}").format(t=ckpt_dir)
      os.system(cmd)
      dirs=os.listdir(model_dir)

      dir_name=[d for d in dirs if not d.startswith(".")]

      dir_name=dir_name[0]

      cmd = ("cp -r {s}/* {d}").format(s=os.path.join(model_dir,dir_name),d=model_dir)
      os.system(cmd)
      cmd = ("rm -rf {t}").format(t=os.path.join(model_dir,dir_name))
      os.system(cmd)
    except Exception as e:
      print("get an exception,but it may worked.")

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
