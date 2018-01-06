PY_INCLUDE_PATH=$(shell python -c "import sysconfig; print sysconfig.get_path('include')")
PY_LIB_PATH=$(shell python -c "import sysconfig; print sysconfig.get_path('stdlib')")

INCLUDE_PATHS = -I$(PY_INCLUDE_PATH) -I$(GGPLIB_PATH)/src/cpp
LIB_PATHS = -L$(PY_LIB_PATH) -L$(GGPLIB_PATH)/src/cpp

AR = ar
RM = rm -f
CPP = g++

CFLAGS = -MMD -MP -g -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -fPIC -std=c++11
LDFLAGS = -g -MMD -MP -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -std=c++11


SRCS = puctnode.cpp pucteval.cpp ggpzero_interface.cpp

CORE_OBJS = $(patsubst %.cpp, %.o, $(SRCS))
OBJS := $(CORE_OBJS)
DEPS=$(SRCS:.cpp=.d)

# Top level
all: $(OBJS) ggpzero_interface.so

fresh : clean all

# Compiles
ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(CORE_OBJS) $(LDFLAGS) $(LIB_PATHS) -pthread -lrt -lpython2.7 -lggplib_cpp -o ggpzero_interface.so

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) $(OBJS) $(DEPS)

realclean : clean
	$(RM) *.so *.log core*

-include $(DEPS)
.PHONY: all fresh clean realclean

