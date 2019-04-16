CXX=g++
INCL_DIRS=-I/usr/include/torch/csrc/api/include -I/usr/include/python3.7m
comm=$(shell ls /opt/cuda/lib 2>&1 >/dev/null; echo $$?)
ifeq ($(comm), 0)
LIB_DIRS=-L/usr/lib/pytorch -L/opt/cuda/lib
LIBS=-lc10 -ltorch -lcaffe2 -lnvrtc -lcuda
else
LIB_DIRS=-L/usr/lib/pytorch
LIBS=-lc10 -ltorch -lcaffe2
endif
BINS=atan
$(BINS): $(BINS).cpp
	$(CXX) $< $(INCL_DIRS) $(LIB_DIRS) $(LIBS) -o $@
.PHONY: clean run
run: $(BINS)
	./$<
clean:
	rm -rf $(BINS)