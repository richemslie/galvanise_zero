include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/build/lib -lk273 -lk273_greenlet -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp -lpython2.7

# needs numpy installed in virtualenv, if not found set THE_PYTHONPATH manually
THE_PYTHONPATH = $(shell python -c "import sys; print sys.exec_prefix")
INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp -I. -I$(THE_PYTHONPATH)/lib/python2.7/site-packages/numpy/core/include

# since dll
CFLAGS += -fPIC

# python specific compile flags
CFLAGS += -Wno-register -Wno-strict-aliasing $(shell python2-config --includes)

SRCS = puct/node.cpp puct/evaluator.cpp player.cpp
#SRCS = minimax/node.cpp minimax/evaluator.cpp

SRCS += gdltransformer.cpp scheduler.cpp selfplay.cpp selfplaymanager.cpp
SRCS += supervisor.cpp ggpzero_interface.cpp

OBJS = $(patsubst %.cpp, %.o, $(SRCS))

DEPS = $(SRCS:.cpp=.d)

# Top level
all: $(OBJS) ggpzero_interface.so
	@echo $(THE_PYTHONPATH)

ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) ggpzero_interface.so $(OBJS) $(DEPS)

-include $(DEPS)
.PHONY: all clean

