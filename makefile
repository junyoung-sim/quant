COMPILER=g++
VERSION=-std=c++11

output: main.o data.o node.o layer.o neural_network.o dqn.o
	$(COMPILER) $(VERSION) main.o data.o node.o layer.o neural_network.o dqn.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

data.o: ./src/data.cpp
	$(COMPILER) $(VERSION) -c ./src/data.cpp

node.o: ./src/node.cpp
	$(COMPILER) $(VERSION) -c ./src/node.cpp

layer.o: ./src/layer.cpp
	$(COMPILER) $(VERSION) -c ./src/layer.cpp

neural_network.o: ./src/neural_network.cpp
	$(COMPILER) $(VERSION) -c ./src/neural_network.cpp

dqn.o: ./src/dqn.cpp
	$(COMPILER) $(VERSION) -c ./src/dqn.cpp
