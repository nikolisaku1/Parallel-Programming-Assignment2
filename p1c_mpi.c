/* Random Restart Hill Climbing using MPI
 *
 * Compile: mpicc -o p1c_mpi p1c_mpi.c -lm
 * Run:     mpiexec -n 4 ./p1c_mpi <numRestarts> <minX> <maxX> <minY> <maxY> <stepSize>
 * Example: mpiexec -n 4 ./p1c_mpi 1000 0 10 0 10 0.01
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

double objectiveFunction(double x, double y) {
    return 50 + 20*x*y - 3*x*x*y - 2*x*y*y;
}

void hillClimb(double minX, double maxX, double minY, double maxY,
               double stepSize, unsigned int *seed,
               double *bestVal, double *bestX, double *bestY)
{
    double cx = minX + ((double)rand_r(seed) / RAND_MAX) * (maxX - minX);
    double cy = minY + ((double)rand_r(seed) / RAND_MAX) * (maxY - minY);
    double cv = objectiveFunction(cx, cy);

    int improved = 1;
    while (improved) {
        improved = 0;
        double dx[] = {-1, 1,  0, 0, -1,  1, -1, 1};
        double dy[] = { 0, 0, -1, 1, -1, -1,  1, 1};
        for (int i = 0; i < 8; i++) {
            double nx = cx + dx[i] * stepSize;
            double ny = cy + dy[i] * stepSize;
            if (nx < minX || nx > maxX || ny < minY || ny > maxY) continue;
            double nv = objectiveFunction(nx, ny);
            if (nv > cv) { cv = nv; cx = nx; cy = ny; improved = 1; }
        }
    }
    *bestVal = cv; *bestX = cx; *bestY = cy;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    totalRestarts;
    double minX, maxX, minY, maxY, stepSize;

    /* Only rank 0 reads input, then broadcasts */
    if (rank == 0) {
        if (argc != 7) {
            fprintf(stderr, "Usage: mpiexec -n <P> %s <numRestarts> <minX> <maxX> <minY> <maxY> <stepSize>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        totalRestarts = atoi(argv[1]);
        minX = atof(argv[2]); maxX = atof(argv[3]);
        minY = atof(argv[4]); maxY = atof(argv[5]);
        stepSize = atof(argv[6]);
    }

    /* Broadcast parameters to all processes */
    MPI_Bcast(&totalRestarts, 1, MPI_INT,    0, MPI_COMM_WORLD);
    double params[5] = {minX, maxX, minY, maxY, stepSize};
    MPI_Bcast(params, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    minX = params[0]; maxX = params[1];
    minY = params[2]; maxY = params[3];
    stepSize = params[4];

    /* Each process handles a chunk of restarts */
    int base  = totalRestarts / size;
    int extra = totalRestarts % size;
    int myRestarts = base + (rank < extra ? 1 : 0);

    unsigned int seed = (unsigned int)(time(NULL)) ^ (unsigned int)(rank * 2654435761u);
    double localBest = -1e300, localX = 0.0, localY = 0.0;

    for (int r = 0; r < myRestarts; r++) {
        double val, x, y;
        hillClimb(minX, maxX, minY, maxY, stepSize, &seed, &val, &x, &y);
        if (val > localBest) { localBest = val; localX = x; localY = y; }
    }

    /* Custom reduction: find process holding the global maximum */
    /* Pack (value, x, y) together for a single reduce */
    double localTriple[3] = {localBest, localX, localY};
    double globalTriple[3];

    /* Use MPI_MAXLOC on the value; we need a custom approach:
       gather all triples to rank 0 and find the best */
    double *allTriples = NULL;
    if (rank == 0) allTriples = malloc(size * 3 * sizeof(double));

    MPI_Gather(localTriple, 3, MPI_DOUBLE, allTriples, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double bestVal = -1e300, bestX = 0.0, bestY = 0.0;
        for (int p = 0; p < size; p++) {
            if (allTriples[p*3] > bestVal) {
                bestVal = allTriples[p*3];
                bestX   = allTriples[p*3+1];
                bestY   = allTriples[p*3+2];
            }
        }
        printf("Best value : %.6f\n", bestVal);
        printf("At (x, y)  : (%.6f, %.6f)\n", bestX, bestY);
        free(allTriples);
    }

    MPI_Finalize();
    return 0;
}
