COM=g++
VER=-std=c++11

output: main.o data.o net.o quant.o checkpoint.o
	$(COM) $(VER) main.o data.o net.o quant.o checkpoint.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp

data.o: ./src/data.cpp
	$(COM) $(VER) -c ./src/data.cpp

net.o: ./src/net.cpp
	$(COM) $(VER) -c ./src/net.cpp

quant.o: ./src/quant.cpp
	$(COM) $(VER) -c ./src/quant.cpp

checkpoint.o: ./src/checkpoint.cpp
	$(COM) $(VER) -c ./src/checkpoint.cpp
