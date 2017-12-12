
PY_FILES := $(shell find $(GGPLEARN_PATH)/src/ggplearn -name '*.py')

flake8:
	flake8 $(PY_FILES)
	pylint $(PY_FILES)
