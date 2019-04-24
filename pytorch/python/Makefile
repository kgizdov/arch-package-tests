TESTS = two_layer_nn_test.py
TESTSCLEAN = $(addsuffix .clean,$(TESTS))

.PHONY: all run clean tests $(TESTS) $(TESTSCLEAN)

tests: $(TESTS)

$(TESTS):
	@./$@

all: tests

run: all

clean: $(TESTSCLEAN)

$(TESTSCLEAN): %.clean:

