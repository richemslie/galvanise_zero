install instructions
====================

Using python 2, keras and tensorflow.  Install two different environments for cpu/gpu.

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


possible todos
--------------

* mypy (sanity)

* break up process_games.py smaller files

* some sort of monte-carlo / iterative deeping player

* self play, retrain and evaluate

* figure out how to use tensorflow (cpu) to predict in c++ (not train)
