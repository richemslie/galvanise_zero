include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/build/lib -lk273 -lk273_greenlet -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp -lpython2.7
INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp -I.

# since dll
CFLAGS += -fPIC

# python specific compile flags
CFLAGS += -Wno-register -Wno-strict-aliasing $(shell python2-config --includes)

SRCS = puct/node.cpp puct/evaluator.cpp
SRCS += gdltransformer.cpp scheduler.cpp selfplay.cpp selfplaymanager.cpp
SRCS += player.cpp supervisor.cpp ggpzero_interface.cpp

# experimental
SRCS += puct2/node.cpp puct2/evaluator.cpp player2.cpp

OBJS = $(patsubst %.cpp, %.o, $(SRCS))
DEPS=$(SRCS:.cpp=.d)

# Top level
all: $(OBJS) ggpzero_interface.so

ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Install
install: $(LIB_NAME)
	$(CP) $(LIB_NAME) $(INSTALL_LIB_PATH)

# Cleans
clean :
	$(RM) $(OBJS) $(DEPS)


-include $(DEPS)
.PHONY: all clean

