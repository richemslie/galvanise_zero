if [ -z "$GGPLIB_PATH" ]; then
    echo "Please set \$GGPLIB_PATH"

else
    echo "Ensuring ggplib is set up..."
    . $GGPLIB_PATH/bin/setup.sh

    export GGPZERO_PATH=`python2 -c "import os.path as p; print p.dirname(p.dirname(p.abspath('$BASH_SOURCE')))"`
    echo "Automatically setting \$GGPZERO_PATH to $GGPZERO_PATH"

    export PYTHONPATH=$GGPZERO_PATH/src:$PYTHONPATH
    export LD_LIBRARY_PATH=$GGPZERO_PATH/src/cpp:$LD_LIBRARY_PATH
    export PATH=$GGPZERO_PATH/bin:$PATH

    cd $GGPZERO_PATH/src/ggpzero
fi
