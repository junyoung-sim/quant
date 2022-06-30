COMPILER=g++
VERSION=-std=c++11

output: main.o neural_network.o rl.o
	$(COMPILER) $(VERSION) main.o neural_network.o rl.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COMPILER) $(VERSION) -c ./src/main.cpp

neural_network.o: ./src/nn/node.cpp ./src/nn/layer.cpp ./src/nn/neural_network.cpp
	$(COMPILER) $(VERSION) -c ./src/nn/node.cpp ./src/nn/layer.cpp ./src/nn/neural_network.cpp

rl.o: ./src/rl/dqn.cpp
	$(COMPILER) $(VERSION) -C ./src/rl/dqn.cpp
