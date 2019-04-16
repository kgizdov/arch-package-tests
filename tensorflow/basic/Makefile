CXX=g++
BINS=basic
INCL_DIR=-I/usr/lib/python3.7/site-packages/tensorflow/include
LIBS=-ltensorflow_cc -ltensorflow_framework
$(BINS): $(BINS).cpp
	$(CXX) $< $(INCL_DIR) $(LIBS) -o $@
.PHONY: clean run
run: $(BINS)
	./$<
clean:
	rm -rf $(BINS)
