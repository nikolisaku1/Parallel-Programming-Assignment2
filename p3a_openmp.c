#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BUFFER_SIZE  10     /* max words in shared buffer */
#define NUM_PRODUCERS 3     /* one file per producer */
#define WORD_LEN      64

/* Producer file list */
const char *producerFiles[NUM_PRODUCERS] = {
    "words_producer0.txt",
    "words_producer1.txt",
    "words_producer2.txt"
};

/* Shared circular buffer */
char   buffer[BUFFER_SIZE][WORD_LEN];
int    head = 0, tail = 0, count = 0;
int    producersDone = 0;          /* how many producers have finished */

omp_lock_t bufLock;

/* Buffer helpers (call only while holding bufLock) */
static inline int bufFull()  { return count == BUFFER_SIZE; }
static inline int bufEmpty() { return count == 0; }

void bufPut(const char *word) {
    strncpy(buffer[tail], word, WORD_LEN - 1);
    buffer[tail][WORD_LEN - 1] = '\0';
    tail = (tail + 1) % BUFFER_SIZE;
    count++;
}

void bufGet(char *word) {
    strncpy(word, buffer[head], WORD_LEN - 1);
    word[WORD_LEN - 1] = '\0';
    head = (head + 1) % BUFFER_SIZE;
    count--;
}

/* Producer */
void producer(int id) {
    char word[WORD_LEN];
    FILE *f = fopen(producerFiles[id], "r");
    if (!f) {
        fprintf(stderr, "Producer %d: cannot open %s\n", id, producerFiles[id]);
    } else {
        while (fscanf(f, "%63s", word) == 1) {
            /* Spin-wait until there is space in the buffer */
            int placed = 0;
            while (!placed) {
                omp_set_lock(&bufLock);
                if (!bufFull()) {
                    bufPut(word);
                    placed = 1;
                    printf("[Producer %d] PUT    \"%s\"  (buffer: %d/%d)\n",
                           id, word, count, BUFFER_SIZE);
                }
                omp_unset_lock(&bufLock);
                if (!placed) {
                    /* Yield without holding the lock */
                    #pragma omp taskyield
                }
            }
        }
        fclose(f);
    }

    omp_set_lock(&bufLock);
    producersDone++;
    omp_unset_lock(&bufLock);
    printf("[Producer %d] finished.\n", id);
}

/* Consumer */
void consumer(int id) {
    char word[WORD_LEN];
    while (1) {
        int got = 0;
        int allDone = 0;

        omp_set_lock(&bufLock);
        if (!bufEmpty()) {
            bufGet(word);
            got = 1;
            printf("[Consumer %d] GOT    \"%s\"  (buffer: %d/%d)\n",
                   id, word, count, BUFFER_SIZE);
        }
        allDone = (producersDone == NUM_PRODUCERS) && bufEmpty();
        omp_unset_lock(&bufLock);

        if (got) {
            /* Tokenization is trivial here since we store one word per slot.
               In a multi-word-line scenario we would call strtok here. */
            printf("[Consumer %d] TOKEN: \"%s\"\n", id, word);
        } else if (allDone) {
            break;
        } else {
            #pragma omp taskyield
        }
    }
    printf("[Consumer %d] finished.\n", id);
}

/* Main */
int main(void) {
    int numConsumers = 2;
    int totalThreads = NUM_PRODUCERS + numConsumers;

    omp_init_lock(&bufLock);
    omp_set_num_threads(totalThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid < NUM_PRODUCERS)
            producer(tid);
        else
            consumer(tid - NUM_PRODUCERS);
    }

    omp_destroy_lock(&bufLock);
    printf("\nAll done.\n");
    return 0;
}
