include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/src/cpp/k273 -lk273 -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp -lpython2.7
INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp -I.

# since dll
CFLAGS += -fPIC

# python specific compile flags
CFLAGS += -Wno-register -Wno-strict-aliasing $(shell python2-config --includes)

SRCS = puct/node.cpp puct/evaluator.cpp
SRCS += gdltransformer.cpp scheduler.cpp selfplay.cpp selfplaymanager.cpp
SRCS += player.cpp supervisor.cpp ggpzero_interface.cpp
SRCS += greenlet/greenlet.cpp greenlet/greenlet-sys.cpp

CORE_OBJS = $(patsubst %.cpp, %.o, $(SRCS))
OBJS := $(CORE_OBJS) greenlet/greenlet-asm.o
DEPS=$(SRCS:.cpp=.d)

# Top level
all: $(OBJS) ggpzero_interface.so

# Compiles
#	$(CPP) -shared $(OBJS) $(LDFLAGS) $(LIB_PATHS) -pthread -lrt -lpython2.7 -lggplib_cpp -o ggpzero_interface.so

ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(LDFLAGS) $(OBJS) $(LIBS) -o $@

greenlet/greenlet-asm.o: greenlet/greenlet-asm.S
	gcc -fPIC -c -o $@ $<

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) $(OBJS) $(DEPS)


-include $(DEPS)
.PHONY: all clean

