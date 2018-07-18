include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/src/cpp/k273 -lk273 -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp
INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp

CFLAGS += -fPIC

SRCS = mcts.cpp worker.cpp interface.cpp

OBJS = $(patsubst %.cpp, %.o, $(SRCS))
DEPS=$(SRCS:.cpp=.d)

# Top level
all: $(OBJS) libgurgehplayer_cpp.so

# Compiles
libgurgehplayer_cpp.so: $(OBJS)
	$(CPP) -shared $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) libgurgehplayer_cpp.so $(OBJS) $(DEPS)

-include $(DEPS)
.PHONY: all clean


