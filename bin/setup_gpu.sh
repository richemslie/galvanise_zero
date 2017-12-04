if [ -z "$GGPLIB_PATH" ]; then
    echo "Please set \$GGPLIB_PATH"

else
    echo "Ensuring ggplib is set up..."
    . $GGPLIB_PATH/bin/setup.sh

    export GGPLEARN_PATH=`python2 -c "import os.path as p; print p.dirname(p.dirname(p.abspath('$BASH_SOURCE')))"`
    echo "Automatically setting \$GGPLEARN_PATH to $GGPLEARN_PATH"

    # to activate python
    . $GGPLEARN_PATH/bin/install/_python2_gpu/bin/activate

    export PYTHONPATH=$GGPLEARN_PATH/src:$PYTHONPATH
    export LD_LIBRARY_PATH=$GGPLEARN_PATH/src/cpp:$LD_LIBRARY_PATH
    export PATH=$GGPLEARN_PATH/bin:$PATH
fi

