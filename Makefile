CFLAGS=-O3 -std=c17

.PHONY: all
all: main
main: ImplementationNaive.c ImplementationSeries.c ImplementationLUT.c SupportingProgram.c
	$(CC) $(CFLAGS) -o $@ $^ -lm -D_POSIX_C_SOURCE=199309L -msse4.1
	rm -f ImplementationLUT.o ImplementationNaive.o ImplementationSeries.o

.PHONY: clean
clean:
	rm -f main