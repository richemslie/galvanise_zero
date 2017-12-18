
PY_FILES := $(shell find $(GGPLEARN_PATH)/src/ggplearn -name '*.py')
TEST_FILES := $(shell find $(GGPLEARN_PATH)/src/test -name '*.py')

all:
	flake8 $(PY_FILES)
	pylint $(PY_FILES)

test:
	flake8 $(TEST_FILES)
	pylint $(TEST_FILES)
