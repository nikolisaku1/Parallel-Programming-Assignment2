#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

/* Objective function */
double objectiveFunction(double x, double y) {
    return 50 + 20*x*y - 3*x*x*y - 2*x*y*y;
}

/* Shared problem parameters */
double g_minX, g_maxX, g_minY, g_maxY, g_stepSize;
int    g_totalRestarts, g_numThreads;

/* Global best (protected by mutex) */
double   g_bestVal = -1e300;
double   g_bestX   = 0.0, g_bestY = 0.0;
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Per-thread argument */
typedef struct {
    int thread_id;
    int restarts;
} ThreadArg;

/* Single hill-climb from a random start */
void hillClimb(double *bestVal, double *bestX, double *bestY, unsigned int *seed) {
    double cx = g_minX + ((double)rand_r(seed) / RAND_MAX) * (g_maxX - g_minX);
    double cy = g_minY + ((double)rand_r(seed) / RAND_MAX) * (g_maxY - g_minY);
    double cv = objectiveFunction(cx, cy);

    int improved = 1;
    while (improved) {
        improved = 0;
        double dx[] = {-1, 1,  0, 0, -1,  1, -1, 1};
        double dy[] = { 0, 0, -1, 1, -1, -1,  1, 1};

        for (int i = 0; i < 8; i++) {
            double nx = cx + dx[i] * g_stepSize;
            double ny = cy + dy[i] * g_stepSize;
            if (nx < g_minX || nx > g_maxX || ny < g_minY || ny > g_maxY)
                continue;
            double nv = objectiveFunction(nx, ny);
            if (nv > cv) { cv = nv; cx = nx; cy = ny; improved = 1; }
        }
    }
    *bestVal = cv; *bestX = cx; *bestY = cy;
}

/* Thread function */
void *threadFunc(void *arg) {
    ThreadArg *ta = (ThreadArg *)arg;
    unsigned int seed = (unsigned int)(time(NULL)) ^ (unsigned int)(ta->thread_id * 1234567);
    double localBest = -1e300, localX = 0.0, localY = 0.0;

    for (int r = 0; r < ta->restarts; r++) {
        double val, x, y;
        hillClimb(&val, &x, &y, &seed);
        if (val > localBest) { localBest = val; localX = x; localY = y; }
    }

    pthread_mutex_lock(&g_mutex);
    if (localBest > g_bestVal) { g_bestVal = localBest; g_bestX = localX; g_bestY = localY; }
    pthread_mutex_unlock(&g_mutex);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <numRestarts> <numThreads> <minX> <maxX> <minY> <maxY> <stepSize>\n", argv[0]);
        return 1;
    }
    g_totalRestarts = atoi(argv[1]);
    g_numThreads    = atoi(argv[2]);
    g_minX = atof(argv[3]); g_maxX = atof(argv[4]);
    g_minY = atof(argv[5]); g_maxY = atof(argv[6]);
    g_stepSize = atof(argv[7]);

    pthread_t *threads = malloc(g_numThreads * sizeof(pthread_t));
    ThreadArg *args    = malloc(g_numThreads * sizeof(ThreadArg));
    int base = g_totalRestarts / g_numThreads, extra = g_totalRestarts % g_numThreads;

    for (int t = 0; t < g_numThreads; t++) {
        args[t].thread_id = t;
        args[t].restarts  = base + (t < extra ? 1 : 0);
        pthread_create(&threads[t], NULL, threadFunc, &args[t]);
    }
    for (int t = 0; t < g_numThreads; t++) pthread_join(threads[t], NULL);

    printf("Best value : %.6f\n", g_bestVal);
    printf("At (x, y)  : (%.6f, %.6f)\n", g_bestX, g_bestY);

    free(threads); free(args);
    pthread_mutex_destroy(&g_mutex);
    return 0;
}
