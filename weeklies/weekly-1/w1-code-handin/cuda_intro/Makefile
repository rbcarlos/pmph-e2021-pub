COMPILER?=nvcc
OPT_FLAGS?=-O3 -arch=compute_35
LDFLAGS?=-lOpenCL

.PHONY: clean all run

default: compile run

compile: wa1-task3.cu 
	$(COMPILER) $(OPT_FLAGS) -o wa1-task3 wa1-task3.cu

run: wa1-task3
	./wa1-task3

clean:
	rm -f wa1-task3
