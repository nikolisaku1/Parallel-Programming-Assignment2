/* Problem: Find all prime numbers up to N (Sieve of Eratosthenes)
 *
 * Serial solution: standard sequential sieve – O(N log log N)
 * Parallel shared-memory: OpenMP parallel sieve
 * Parallel distributed: MPI segmented sieve
 *
 * ── SERIAL DESCRIPTION ───────────────────────────────────────────────
 * The Sieve of Eratosthenes marks multiples of each prime p starting
 * from p^2 as composite. After processing all p <= sqrt(N), remaining
 * unmarked indices are prime.
 *
 * Compile (serial):  gcc -DN=10000000 -o p4_serial p4_prime_sieve.c -lm
 * Compile (OpenMP):  gcc -fopenmp -DOPENMP -DN=10000000 -o p4_omp p4_prime_sieve.c -lm
 * Compile (MPI):     mpicc -DMPI_VERSION -DN=10000000 -o p4_mpi p4_prime_sieve.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef N
#define N 10000000
#endif

/* SERIAL SIEVE */

#if !defined(OPENMP) && !defined(MPI_VERSION)

int main(void) {
    char *composite = calloc(N + 1, 1);  /* 0 = prime, 1 = composite */
    composite[0] = composite[1] = 1;

    int sqrtN = (int)sqrt((double)N);
    for (int p = 2; p <= sqrtN; p++) {
        if (!composite[p]) {
            for (int m = p * p; m <= N; m += p)
                composite[m] = 1;
        }
    }

    long long cnt = 0;
    for (int i = 2; i <= N; i++) if (!composite[i]) cnt++;
    printf("[Serial] Primes up to %d: %lld\n", N, cnt);
    free(composite);
    return 0;
}

/* OPENMP PARALLEL SIEVE
 * ══════════════════════════════════════════════════════════════════════
 * Strategy: marking phase is data-parallel — each iteration of the
 * inner loop (marking multiples) is independent. We parallelise the
 * outer loop over primes p.  The composites array is written without
 * conflict because each thread writes to disjoint (or harmlessly
 * duplicate) locations.
 * ══════════════════════════════════════════════════════════════════════ */
#elif defined(OPENMP)
#include <omp.h>

int main(void) {
    char *composite = calloc(N + 1, 1);
    composite[0] = composite[1] = 1;
    int sqrtN = (int)sqrt((double)N);

    /* Phase 1: sieve up to sqrt(N) sequentially (small, fast) */
    for (int p = 2; p <= sqrtN; p++) {
        if (!composite[p]) {
            for (int m = p * p; m <= sqrtN; m += p)
                composite[m] = 1;
        }
    }

    /* Phase 2: collect small primes */
    int smallPrimes[10000]; int spCount = 0;
    for (int p = 2; p <= sqrtN; p++)
        if (!composite[p]) smallPrimes[spCount++] = p;

    /* Phase 3: parallel marking of the rest of the array */
    #pragma omp parallel for schedule(dynamic, 64)
    for (int pi = 0; pi < spCount; pi++) {
        int p = smallPrimes[pi];
        /* Find first multiple of p that is > sqrtN */
        long long start = ((long long)sqrtN / p + 1) * p;
        for (long long m = start; m <= N; m += p)
            composite[m] = 1;
    }

    long long cnt = 0;
    #pragma omp parallel for reduction(+:cnt)
    for (int i = 2; i <= N; i++) if (!composite[i]) cnt++;

    printf("[OpenMP] Primes up to %d: %lld (threads: %d)\n",
           N, cnt, omp_get_max_threads());
    free(composite);
    return 0;
}

/* MPI SEGMENTED SIEVE
 * ══════════════════════════════════════════════════════════════════════
 * Strategy: rank 0 sieves [2, sqrt(N)] sequentially and broadcasts
 * the list of small primes.  The range [sqrt(N)+1 .. N] is divided
 * into contiguous segments; each process sieves its own segment using
 * the shared small-prime list.  Finally, each process reports its
 * prime count and rank 0 sums them.
 * ══════════════════════════════════════════════════════════════════════ */
#elif defined(MPI_VERSION)
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sqrtN = (int)sqrt((double)N);

    /* Phase 1: rank 0 sieves [0..sqrtN] */
    int *smallPrimes = malloc((sqrtN + 1) * sizeof(int));
    int  spCount = 0;

    if (rank == 0) {
        char *comp0 = calloc(sqrtN + 1, 1);
        comp0[0] = comp0[1] = 1;
        for (int p = 2; p <= sqrtN; p++) {
            if (!comp0[p]) {
                smallPrimes[spCount++] = p;
                for (int m = p * p; m <= sqrtN; m += p) comp0[m] = 1;
            }
        }
        free(comp0);
    }

    /* Phase 2: broadcast small primes */
    MPI_Bcast(&spCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(smallPrimes, spCount, MPI_INT, 0, MPI_COMM_WORLD);

    /* Phase 3: each process sieves its segment */
    long long total   = (long long)N - sqrtN;   /* elements from sqrtN+1 to N */
    long long segLen  = (total + size - 1) / size;
    long long myStart = sqrtN + 1 + (long long)rank * segLen;
    long long myEnd   = myStart + segLen - 1;
    if (myEnd > N) myEnd = N;
    if (myStart > N) myStart = myEnd + 1; /* empty segment */

    long long myLen = (myStart <= myEnd) ? (myEnd - myStart + 1) : 0;
    char *seg = calloc(myLen + 1, 1);

    for (int pi = 0; pi < spCount; pi++) {
        long long p = smallPrimes[pi];
        /* First multiple of p >= myStart */
        long long first = ((myStart + p - 1) / p) * p;
        if (first == p) first += p;   /* skip p itself */
        for (long long m = first; m <= myEnd; m += p)
            seg[m - myStart] = 1;
    }

    long long localCnt = 0;
    for (long long i = 0; i < myLen; i++) if (!seg[i]) localCnt++;
    if (rank == 0) localCnt += spCount;  /* add small primes */

    long long totalPrimes = 0;
    MPI_Reduce(&localCnt, &totalPrimes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("[MPI] Primes up to %d: %lld (processes: %d)\n", N, totalPrimes, size);

    free(seg); free(smallPrimes);
    MPI_Finalize();
    return 0;
}
#endif
