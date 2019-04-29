.PHONY: all run clean
CC = nvcc
LIBS = -lcudnn
BINS = cudnnver
CXXFLAGS = -O3 --compiler-options -Wall --compiler-options -Wextra

all: $(BINS)

%: %.cpp
	@if [ -f /usr/include/cudnn.h ]; then \
	$(CC) -o $@ $^ $(LIBS) $(CXXFLAGS) ;\
	else \
	echo "cuDNN missing. Skipping test."; \
	fi

run: all
	@if [ -f $(BINS) ]; then \
	./$(BINS) ;\
        fi

clean:
	rm -rf $(BINS)
