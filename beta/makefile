COM=g++
VER=-std=c++2a

output: main.o data.o
	$(COM) $(VER) main.o data.o -o exec
	rm *.o

main.o: ./src/main.cpp
	$(COM) $(VER) -c ./src/main.cpp

data.o: ./src/data.cpp
	$(COM) $(VER) -c ./src/data.cpp