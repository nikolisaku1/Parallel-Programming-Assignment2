/* Random Restart Hill Climbing using OpenMP
 *
 * Compile: gcc -fopenmp -o p1b_openmp p1b_openmp.c -lm
 * Run:     ./p1b_openmp <numRestarts> <numThreads> <minX> <maxX> <minY> <maxY> <stepSize>
 * Example: ./p1b_openmp 1000 4 0 10 0 10 0.01
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
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
    if (argc != 8) {
        fprintf(stderr, "Usage: %s <numRestarts> <numThreads> <minX> <maxX> <minY> <maxY> <stepSize>\n", argv[0]);
        return 1;
    }
    int    totalRestarts = atoi(argv[1]);
    int    numThreads    = atoi(argv[2]);
    double minX = atof(argv[3]), maxX = atof(argv[4]);
    double minY = atof(argv[5]), maxY = atof(argv[6]);
    double stepSize = atof(argv[7]);

    double globalBest = -1e300, globalX = 0.0, globalY = 0.0;

    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)(time(NULL)) ^ (unsigned int)(tid * 2654435761u);
        double localBest = -1e300, localX = 0.0, localY = 0.0;

        #pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < totalRestarts; r++) {
            double val, x, y;
            hillClimb(minX, maxX, minY, maxY, stepSize, &seed, &val, &x, &y);
            if (val > localBest) { localBest = val; localX = x; localY = y; }
        }

        #pragma omp critical
        {
            if (localBest > globalBest) {
                globalBest = localBest;
                globalX = localX;
                globalY = localY;
            }
        }
    }

    printf("Best value : %.6f\n", globalBest);
    printf("At (x, y)  : (%.6f, %.6f)\n", globalX, globalY);
    return 0;
}
