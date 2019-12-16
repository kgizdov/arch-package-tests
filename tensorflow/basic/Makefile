CXX=g++
BINS=basic
INCL_DIR=$(shell pkg-config tensorflow_cc --cflags)
LIBS=$(shell pkg-config tensorflow_cc --libs)
$(BINS): $(BINS).cpp
	$(CXX) $< $(INCL_DIR) $(LIBS) -o $@
.PHONY: clean run
run: $(BINS)
	./$<
clean:
	rm -rf $(BINS)
