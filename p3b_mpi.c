 /* Producer-Consumer using MPI
 *
 * Rank 0            : master / coordinator
 * Ranks 1..P        : producers (read files, send words to master buffer)
 * Ranks P+1..end    : consumers (receive words from master, tokenize, display)
 *
 * With the default layout (5 processes total):
 *   rank 0  -> master buffer
 *   ranks 1-3 -> producers (3 files)
 *   rank 4  -> consumer
 *
 * Compile: mpicc -o p3b_mpi p3b_mpi.c
 * Run:     mpiexec -n 5 ./p3b_mpi
 *
 * Tags:
 *   TAG_WORD  (10) : producer sends a word to master
 *   TAG_DONE  (11) : producer signals end of file
 *   TAG_TOKEN (20) : master sends a word to a consumer
 *   TAG_STOP  (21) : master tells consumer to stop
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define BUFFER_SIZE   10
#define WORD_LEN      64
#define NUM_PRODUCERS 3
#define TAG_WORD      10
#define TAG_DONE      11
#define TAG_TOKEN     20
#define TAG_STOP      21

const char *producerFiles[NUM_PRODUCERS] = {
    "words_producer0.txt",
    "words_producer1.txt",
    "words_producer2.txt"
};

/* Simple array-based queue (for master) */
char   queue[BUFFER_SIZE][WORD_LEN];
int    qHead = 0, qTail = 0, qCount = 0;
static inline int qFull()  { return qCount == BUFFER_SIZE; }
static inline int qEmpty() { return qCount == 0; }
void   qPut(const char *w) { strncpy(queue[qTail],w,WORD_LEN-1); queue[qTail][WORD_LEN-1]='\0'; qTail=(qTail+1)%BUFFER_SIZE; qCount++; }
void   qGet(char *w)       { strncpy(w,queue[qHead],WORD_LEN-1); w[WORD_LEN-1]='\0'; qHead=(qHead+1)%BUFFER_SIZE; qCount--; }

/* Master (rank 0) */
void master(int size) {
    int numProducers = NUM_PRODUCERS;             /* ranks 1..numProducers */
    int firstConsumer = numProducers + 1;
    int numConsumers  = size - firstConsumer;     /* remaining ranks */

    int producersDone = 0;
    MPI_Status status;
    char word[WORD_LEN];
    int nextConsumer = firstConsumer;

    printf("[Master] started. Producers: %d, Consumers: %d\n",
           numProducers, numConsumers);

    while (1) {
        /* Try to drain queue to available consumers */
        while (!qEmpty()) {
            qGet(word);
            MPI_Send(word, WORD_LEN, MPI_CHAR, nextConsumer, TAG_TOKEN, MPI_COMM_WORLD);
            printf("[Master] dispatched \"%s\" to consumer rank %d\n", word, nextConsumer);
            nextConsumer = firstConsumer + (nextConsumer - firstConsumer + 1) % numConsumers;
        }

        if (producersDone == numProducers) break;

        /* Check for incoming word or done signal (non-blocking probe) */
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            MPI_Recv(word, WORD_LEN, MPI_CHAR, status.MPI_SOURCE,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_WORD) {
                if (!qFull()) {
                    qPut(word);
                    printf("[Master] buffered \"%s\" (count=%d)\n", word, qCount);
                } else {
                    /* Buffer full: re-send back to producer as back-pressure
                       (simple approach: producer will retry via blocking send) */
                    /* In practice, we just push anyway — producers use blocking
                       send and master loops fast enough */
                    qPut(word); /* overwrite warning: shouldn't happen in practice */
                }
            } else if (status.MPI_TAG == TAG_DONE) {
                producersDone++;
                printf("[Master] producer rank %d done (%d/%d)\n",
                       status.MPI_SOURCE, producersDone, numProducers);
            }
        }
    }

    /* Flush remaining queue */
    while (!qEmpty()) {
        qGet(word);
        MPI_Send(word, WORD_LEN, MPI_CHAR, nextConsumer, TAG_TOKEN, MPI_COMM_WORLD);
        nextConsumer = firstConsumer + (nextConsumer - firstConsumer + 1) % numConsumers;
    }

    /* Tell consumers to stop */
    for (int c = firstConsumer; c < size; c++)
        MPI_Send("", WORD_LEN, MPI_CHAR, c, TAG_STOP, MPI_COMM_WORLD);

    printf("[Master] all done.\n");
}

/* Producer */
void producer(int rank) {
    int fileIdx = rank - 1;   /* ranks 1,2,3 -> files 0,1,2 */
    char word[WORD_LEN];

    FILE *f = fopen(producerFiles[fileIdx], "r");
    if (!f) {
        fprintf(stderr, "[Producer %d] cannot open %s\n", rank, producerFiles[fileIdx]);
    } else {
        while (fscanf(f, "%63s", word) == 1) {
            MPI_Send(word, WORD_LEN, MPI_CHAR, 0, TAG_WORD, MPI_COMM_WORLD);
        }
        fclose(f);
    }
    /* Signal done */
    MPI_Send("", WORD_LEN, MPI_CHAR, 0, TAG_DONE, MPI_COMM_WORLD);
    printf("[Producer %d] finished.\n", rank);
}

/* Consumer */
void consumer(int rank) {
    char word[WORD_LEN];
    MPI_Status status;
    while (1) {
        MPI_Recv(word, WORD_LEN, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) break;
        /* Tokenize (trivial: each message is already one word) */
        printf("[Consumer %d] TOKEN: \"%s\"\n", rank, word);
    }
    printf("[Consumer %d] finished.\n", rank);
}

/* Main */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < NUM_PRODUCERS + 2) {
        if (rank == 0)
            fprintf(stderr, "Need at least %d processes (1 master + %d producers + 1 consumer)\n",
                    NUM_PRODUCERS + 2, NUM_PRODUCERS);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
        master(size);
    else if (rank <= NUM_PRODUCERS)
        producer(rank);
    else
        consumer(rank);

    MPI_Finalize();
    return 0;
}
