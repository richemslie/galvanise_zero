include $(K273_PATH)/src/cpp/Makefile.in

LIBS = -L $(K273_PATH)/src/cpp/k273 -lk273 -L$(GGPLIB_PATH)/src/cpp -lggplib_cpp

INCLUDE_PATHS += -I $(K273_PATH)/3rd/cpp
INCLUDE_PATHS += -I $(K273_PATH)/3rd/cpp/fmt-4.1.0

INCLUDE_PATHS += -I $(GGPLIB_PATH)/src/cpp

CFLAGS += -fPIC

# XXX fix up so interface/lib/catch2 make sense

COMMON_SRCS = draughts_desc.cpp draughts_board.cpp draughts_sm.cpp draughts_init.cpp
COMMON_SRCS += baduk_desc.cpp baduk_init.cpp baduk_board.cpp baduk_sm.cpp
COMMON_OBJS := $(patsubst %.cpp, %.o, $(COMMON_SRCS))

LIB_SRCS = interface.cpp
LIB_OBJS := $(patsubst %.cpp, %.o, $(LIB_SRCS))

CATCH2_BIN = catch2.bin
CATCH2_SRCS = catch2_runner.cpp draughts_test.cpp baduk_test.cpp
CATCH2_OBJS = $(patsubst %.cpp, %.o, $(CATCH2_SRCS))

OBJS = $(COMMON_OBJS) $(LIB_OBJS) $(CATCH2_OBJS)
DEPS = $(OBJS:.o=.d)

# Top level
all: $(OBJS) $(CATCH2_BIN) libgzero_games_cpp.so

$(CATCH2_BIN): $(COMMON_OBJS) $(CATCH2_OBJS)
	$(CPP) $(LDFLAGS) $(COMMON_OBJS) $(CATCH2_OBJS) $(LIBS) -o $@

libgzero_games_cpp.so: $(COMMON_OBJS) $(LIB_OBJS)
	$(CPP) -shared $(LDFLAGS) $(COMMON_OBJS) $(LIB_OBJS) $(LIBS) -o $@

%.o : %.cpp
	$(CPP) $(INCLUDE_PATHS) $(CFLAGS) -c -o $@ $<

# Cleans
clean :
	$(RM) $(CATCH2_BIN) libgzero_games_cpp.so $(OBJS) $(DEPS)

-include $(DEPS)
.PHONY: all clean
