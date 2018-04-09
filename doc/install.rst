There are no instructions - left this way since project is in fluid state.

But here is quick sense of what to do.

1.  First install ggpzero (which also means installing k273).
2.  init the environments (. bin/setup.sh)
3.  cd src/cpp && make
4.  create a virtual python environment


pypy versus cpython
-------------------
ggplib it is recommended to use pypy - as provides fastest access to statemachine (but this really
matters for simplest of games, like c4/ttt).  For ggp-zero we use python2.7.  Some games will take
ages to optimise the propnet in python2.7, but once created will be cached and then there is no speed
difference.  I'd recommend creating/caching the propnet of games you are interested in with pypy,
then switch to python2.7.


virtual environment
-------------------

You'll need to do this twice if want to support CPU and GPU:

1. Follow [tensorflow instructions](https://www.tensorflow.org/install/install_linux) to install
python2.7 in a virtual environment.

2. Activate virtualenv.  Check tensorflow works (whether use CPU/GPU, is up to you -
the whole self learning environment is optimised to do batching on GPU - so it might be a bit
forlorn to use ggp-zero for training without a decent GPU).

3. install python packages.

.. code-block:: shell

   pip install requirements.txt


other
-----

training uses client/server model.  The server can run on install without GPU.  clients can be
remote or local to the machine.  For examples of config see repo [gzero_models](https://github.com/richemslie/gzero_data/).

.. code-block:: shell

   cd src/ggpzero/distributed
   python server.py <conf>
   python worker.py <conf>


5.  Running a model :

.. code-block:: shell

   cd src/ggpzero/player
   python puctplayer <port> <model gen> <evaluations*100>

