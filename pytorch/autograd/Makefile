CXX=g++
PYVER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
USRINCLD=/usr/include
INCL_DIRS=-I$(USRINCLD)/torch/csrc/api/include -I$(USRINCLD)/python$(PYVER) -I$(USRINCLD)/python$(PYVER)m
comm=$(shell ls /opt/cuda/lib 2>&1 >/dev/null; echo $$?)
ifeq ($(comm), 0)
LIB_DIRS=-L/usr/lib/pytorch -L/opt/cuda/lib
LIBS=-lc10 -ltorch -lnvrtc -lcuda
else
LIB_DIRS=-L/usr/lib/pytorch
LIBS=-lc10 -ltorch -lcaffe2
endif
BINS=autograd
$(BINS): $(BINS).cpp
	$(CXX) $< $(INCL_DIRS) $(LIB_DIRS) $(LIBS) -o $@
.PHONY: clean run
run: $(BINS)
	./$<
clean:
	rm -rf $(BINS)
