TESTS = keras_test.py tensorflow_test.py
FILES = test.h5
TESTSCLEAN = $(addsuffix .clean,$(TESTS))
FILESCLEAN = $(addsuffix .clean,$(FILES))

.PHONY: all run clean tests $(TESTS) $(TESTSCLEAN) $(FILESCLEAN)

tests: $(TESTS)

$(TESTS):
	@./$@

test.h5: tensorflow_test.py

keras_test.py: test.h5

all: tests

run: all

clean: $(FILESCLEAN)

$(FILESCLEAN): %.clean:
	rm -rf $*
