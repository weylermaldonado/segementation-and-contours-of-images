all:
	g++ -std=c++11 contornos.cpp `pkg-config --libs --cflags opencv` -lfftw3 -o contornos.o

clean:
	rm -rf o contornos.o