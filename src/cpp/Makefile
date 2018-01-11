PY_INCLUDE_PATH=$(shell python -c "import sysconfig; print sysconfig.get_path('include')")
PY_LIB_PATH=$(shell python -c "import sysconfig; print sysconfig.get_path('stdlib')")

INCLUDE_PATHS = -I$(PY_INCLUDE_PATH) -I$(GGPLIB_PATH)/src/cpp
LIB_PATHS = -L$(PY_LIB_PATH) -L$(GGPLIB_PATH)/src/cpp

AR = ar
RM = rm -f
CPP = g++

CFLAGS = -MMD -MP -g -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -fPIC -std=c++11
LDFLAGS = -g -MMD -MP -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -std=c++11


SRCS  = puct/puctnode.cpp puct/pucteval.cpp
SECS += bases.cpp dummysupervisor.cpp supervisor.cpp selfplay.cpp
SRCS += ggpzero_interface.cpp
SRCS += greenlet/greenlet.cpp greenlet/greenlet-sys.cpp greenlet/mocktest.cpp

CORE_OBJS = $(patsubst %.cpp, %.o, $(SRCS))
OBJS := $(CORE_OBJS) greenlet/greenlet-asm.o
DEPS=$(SRCS:.cpp=.d)

# Top level
all: $(OBJS) ggpzero_interface.so

fresh : clean all

# Compiles
ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(OBJS) $(LDFLAGS) $(LIB_PATHS) -pthread -lrt -lpython2.7 -lggplib_cpp -o ggpzero_interface.so

greenlet/greenlet-asm.o: greenlet/greenlet-asm.S
	gcc -fPIC -c -o $@ $<

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) $(OBJS) $(DEPS)

realclean : clean
	$(RM) *.so *.log core*

-include $(DEPS)
.PHONY: all fresh clean realclean

