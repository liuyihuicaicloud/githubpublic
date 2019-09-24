# from https://github.com/kubeflow/tf-operator/blob/master/examples/distribution_strategy/keras_model_to_estimator.py
# tensorflow-1.12
# usage: python estimator.py '/model-path'

"""An example of training Keras model with multi-worker strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
import time


def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(100)
  dataset = dataset.batch(32)
  return dataset


def main(args):
#   if len(args) < 2:
#     print('You must specify model_dir for checkpoints such as'
#           ' /tmp/tfkeras_example/.')
#     return

  print("sleep 5s......")
  time.sleep(5)
  model_dir = "args[1]"
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
          train_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(
              num_gpus_per_worker=0),
          eval_distribute=tf.contrib.distribute.MirroredStrategy(
              num_gpus_per_worker=0)))
  keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, config=config)

  # Train and evaluate the model. Evaluation will be skipped if there is not an
  # "evaluator" job in the cluster.
  tf.estimator.train_and_evaluate(
      keras_estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(argv=sys.argv)
