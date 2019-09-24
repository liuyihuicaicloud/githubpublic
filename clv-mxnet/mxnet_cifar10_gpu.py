# usage: python3 mxnet_cifar10.py --num_epochs=10 --dataset_path='./data/cifar10_train.rec'
import mxnet as mx
import logging
import argparse
import os

parser = argparse.ArgumentParser(description='Process args for evaluation')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--num_epochs", type=int, default=10, help="The number of epochs as in training.")
parser.add_argument("--dataset_path", type=str, default='./cifar10_train.rec', 
                    help="The dataset_path  as in training.")

parser.add_argument("--model_path", type=str, default='./models', 
                    help="Path to save model.")

parser.add_argument("--model_name", type=str, default='model_onnx',
                    help='Name of saved model.")

FLAGS = parser.parse_args()

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
# ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
# default use cpu
ctx = mx.gpu()

train_iter = mx.io.ImageRecordIter(
  path_imgrec=FLAGS.dataset_path, data_name="data", label_name="softmax_label", 
  batch_size=FLAGS.batch_size, data_shape=(3,28,28), shuffle=True)

#valid_iter = mx.io.ImageRecordIter(
#  path_imgrec="./data/cifar10_val.rec", data_name="data", label_name="softmax_label", 
#  batch_size=batch, data_shape=(3,28,28))

data = mx.sym.var('data')
# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
data = mx.sym.flatten(data=data)

# The first fully-connected layer and the corresponding activation function
fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

# The second fully-connected layer and the corresponding activation function
fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, act_type="relu")

# MNIST has 10 classes
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
# Softmax with cross entropy loss
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on compute context
mlp_model = mx.mod.Module(symbol=mlp, context=ctx)
mlp_model.fit(train_iter,  # train data
              # eval_data=valid_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(FLAGS.batch_size, 100), # output progress for each 100 data batches
              num_epoch=FLAGS.num_epochs)  # train for at most 10 dataset passes
mlp_model.save_params(os.path.join(FLAGS.model_path,FLAGS.model_name))
