CXX=g++
BINS=basic
INCL_DIR=$(shell pkg-config tensorflow --cflags)
LIBS=-ltensorflow_cc -ltensorflow_framework
$(BINS): $(BINS).cpp
	$(CXX) $< $(INCL_DIR) $(LIBS) -o $@
.PHONY: clean run
run: $(BINS)
	./$<
clean:
	rm -rf $(BINS)
