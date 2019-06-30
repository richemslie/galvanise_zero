include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/build/lib -lk273 -lk273_greenlet -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp -lpython2.7
INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp -I.

# since dll
CFLAGS += -fPIC

# python specific compile flags
CFLAGS += -Wno-register -Wno-strict-aliasing $(shell python2-config --includes)

SRCS = puct/node.cpp puct/evaluator.cpp player.cpp
SRCS += puct2/node.cpp puct2/evaluator.cpp player2.cpp

SRCS += gdltransformer.cpp scheduler.cpp selfplay.cpp selfplaymanager.cpp
SRCS += supervisor.cpp ggpzero_interface.cpp

EXT_SRCS = external/draughts_desc.cpp external/draughts_board.cpp external/draughts_sm.cpp external/draughts_init.cpp external/interface.cpp


OBJS = $(patsubst %.cpp, %.o, $(SRCS))
EXT_OBJS = $(patsubst %.cpp, %.o, $(EXT_SRCS))

DEPS = $(SRCS:.cpp=.d)
DEPS += $(E_SRCS:.cpp=.d)

# Top level
all: $(OBJS) $(EXT_OBJS) ggpzero_interface.so libgzero_games_cpp.so

ggpzero_interface.so: $(OBJS)
	$(CPP) -shared $(LDFLAGS) $(OBJS) $(LIBS) -o $@

libgzero_games_cpp.so: $(EXT_OBJS)
	$(CPP) -shared $(LDFLAGS) $(EXT_OBJS) $(LIBS) -o $@

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) $(OBJS) $(EXT_OBJS) $(DEPS)

-include $(DEPS)
.PHONY: all clean

