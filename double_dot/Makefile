CC = icpc
CPPFLAGS = -g -Wall -Wextra -mkl -qopenmp

.PHONY: all clean

all: dot

dot: dot.cpp
	$(CC) $(CPPFLAGS) $^ -o $@

clean: 
	rm -rf dot
