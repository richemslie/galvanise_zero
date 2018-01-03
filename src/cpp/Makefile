INCLUDE_PATHS = -I/usr/include/python2.7 -I$(GGPLIB_PATH)/src/cpp
LIB_PATHS = -L/usr/lib/python2.7 -L$(GGPLIB_PATH)/src/cpp

AR = ar
RM = rm -f
CPP = g++

CFLAGS = -MMD -MP -g -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -fPIC -std=c++11
LDFLAGS = -g -MMD -MP -Wall -Werror -Woverloaded-virtual -Wnon-virtual-dtor -fno-strict-aliasing -O2 -std=c++11


SRCS = ggpzero_interface.cpp

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

