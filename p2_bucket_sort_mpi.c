/* Parallel Bucket Sort using MPI
 *
 * Array: 1000 random integers in [1, 250]
 * Each worker process sorts its own bucket, master collects and prints.
 *
 * Compile: mpicc -o p2_bucket_sort_mpi p2_bucket_sort_mpi.c
 * Run:     mpiexec -n 5 ./p2_bucket_sort_mpi
 *          (use any number of worker processes up to 100)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N          1000
#define MIN_VAL    1
#define MAX_VAL    250

/* Simple insertion sort for small buckets */
void insertionSort(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Number of workers (all processes participate) */
    int numWorkers = size;

    int *data = NULL;

    /* Rank 0: generate data and distribute */
    if (rank == 0) {
        data = malloc(N * sizeof(int));
        srand((unsigned int)time(NULL));
        printf("Generated array (first 20): ");
        for (int i = 0; i < N; i++) {
            data[i] = MIN_VAL + rand() % (MAX_VAL - MIN_VAL + 1);
            if (i < 20) printf("%d ", data[i]);
        }
        printf("...\n");
    }

    /* Broadcast data to all processes */
    /* (Alternative: scatter bucket boundaries, then send only relevant data) */
    int *allData = malloc(N * sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < N; i++) allData[i] = data[i];
    }
    MPI_Bcast(allData, N, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each process determines its value range */
    int range      = MAX_VAL - MIN_VAL + 1;   /* 250 */
    int bucketSize = (range + numWorkers - 1) / numWorkers;
    int myLow  = MIN_VAL + rank * bucketSize;
    int myHigh = myLow + bucketSize - 1;
    if (myHigh > MAX_VAL) myHigh = MAX_VAL;

    /* Collect elements belonging to this bucket */
    int *bucket = malloc(N * sizeof(int));
    int  cnt    = 0;
    for (int i = 0; i < N; i++) {
        if (allData[i] >= myLow && allData[i] <= myHigh)
            bucket[cnt++] = allData[i];
    }

    /* Sort local bucket */
    insertionSort(bucket, cnt);

    /* Gather sorted buckets back to rank 0 */
    /* First, tell rank 0 how many elements each process sorted */
    int *counts = NULL;
    if (rank == 0) counts = malloc(numWorkers * sizeof(int));
    MPI_Gather(&cnt, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute displacements for Gatherv */
    int *displs = NULL;
    int *sorted = NULL;
    if (rank == 0) {
        displs = malloc(numWorkers * sizeof(int));
        displs[0] = 0;
        for (int p = 1; p < numWorkers; p++)
            displs[p] = displs[p-1] + counts[p-1];
        sorted = malloc(N * sizeof(int));
    }

    MPI_Gatherv(bucket, cnt, MPI_INT,
                sorted, counts, displs,
                MPI_INT, 0, MPI_COMM_WORLD);

    /* Rank 0 prints result */
    if (rank == 0) {
        printf("\nSorted array (%d elements):\n", N);
        for (int i = 0; i < N; i++) {
            printf("%d ", sorted[i]);
            if ((i+1) % 25 == 0) printf("\n");
        }
        printf("\nSort complete.\n");
        free(sorted); free(counts); free(displs); free(data);
    }

    free(bucket); free(allData);
    MPI_Finalize();
    return 0;
}
