CC=gcc
CFLAGS = -g -O3 -Wall -std=c99
LDLIBS = -lm

all: classifier 

%.o: %.c
	$(CC) -o $@ -c $^ $(CFLAGS)

classifier: knn.o classifier.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDLIBS)

datasets: datasets.tgz
	tar xvzf datasets.tgz

.PHONY: clean all

clean:	
	rm -rf *.dSYM classifier *.o
