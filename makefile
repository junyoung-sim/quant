COMPILER=g++
VERSION=-std=c++11

output: main.o node.o layer.o neural_network.o checkpoint.o data.o quant.o
	$(COMPILER) $(VERSION) main.o node.o layer.o neural_network.o checkpoint.o data.o quant.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

node.o: ./src/node.cpp
	$(COMPILER) $(VERSION) -c ./src/node.cpp

layer.o: ./src/layer.cpp
	$(COMPILER) $(VERSION) -c ./src/layer.cpp

neural_network.o: ./src/neural_network.cpp
	$(COMPILER) $(VERSION) -c ./src/neural_network.cpp

checkpoint.o: ./src/checkpoint.cpp
	$(COMPILER) $(VERSION) -c ./src/checkpoint.cpp

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

quant.o: ./src/quant.cpp
	$(COMPILER) $(VERSION) -c ./src/quant.cpp
