
install instructions
====================

Using python 2, keras and tensorflow.  Install two different environments.

.. code-block:: shell

   export GGPLEARN_PATH=`pwd`

   # first do cpu
   virtualenv -p /usr/bin/python2 --system-site-packages $GGPLEARN_PATH/bin/install/_python2_cpu
   . $GGPLEARN_PATH/bin/install/_python2_cpu/bin/activate

   pip install twisted pytest cffi
   pip install --upgrade tensorflow
   pip install keras h5py

   # then do gpu
   virtualenv -p /usr/bin/python2 --system-site-packages $GGPLEARN_PATH/bin/install/_python2_gpu
   . $GGPLEARN_PATH/bin/install/_python2_gpu/bin/activate

   pip install twisted pytest cffi
   pip install --upgrade tensorflow-gpu
   pip install keras h5py


Using tensorflow with single cpu
--------------------------------
.. code-block:: python

import tensorflow as tf

# XXX TEST THIS
conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)

sess = tf.Session()
with sess.as_default():
     print tf.constant(42).eval()


possible todos
--------------

* mypy (sanity)

* better player for generating data (don't know, rather just try and improve via self play)

* figure out how to use tensorflow (cpu) to predict in c++ (not train)

* break up process_games.py smaller files

* try different networks

* more layers

    * resnet

* players for self play - start simple and work up

    * a minimax player
    * some sort of monte-carlo, iterative deeping player

* performance test

  * especially cpu/gpu

