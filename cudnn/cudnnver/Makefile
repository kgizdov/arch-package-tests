.PHONY: all run clean
CC = nvcc
LIBS = -lcudnn
BINS = cudnnver
CXXFLAGS = -O3 --compiler-options -Wall --compiler-options -Wextra
all: $(BINS)

%: %.cpp
	$(CC) -o $@ $^ $(LIBS) $(CXXFLAGS)

run: all
	./$(BINS)

clean:
	rm -rf $(BINS)
