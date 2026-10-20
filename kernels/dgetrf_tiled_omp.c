/**
 *
 * @file dgetrf_tiled_omp.c
 *
 * @brief OpenMP task-based tile implementation of the dgetrf.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"

static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }
static inline int min_int(int a, int b) { return (a < b) ? a : b; }

static inline int tile_m(int M, int b, int MT, int ti)
{
    return (ti == MT - 1) ? (M - ti * b) : b;
}

static inline int tile_n(int N, int b, int NT, int tj)
{
    return (tj == NT - 1) ? (N - tj * b) : b;
}

static inline double *tile(double **A, int MT, int ti, int tj)
{
    return A[ MT * tj + ti ];
}

int
dgetrf_tiled_omp( CBLAS_LAYOUT layout, int M, int N, int b, double **A )
{
    assert(layout == CblasColMajor);

    const int MT = ceil_div(M, b);
    const int NT = ceil_div(N, b);
    const int KT = min_int(MT, NT);

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int k = 0; k < KT; ++k) {

                /* Diagonal tile size */
                const int mk = tile_m(M, b, MT, k);
                const int nk = tile_n(N, b, NT, k);
                const int kk = min_int(mk, nk);

                double *Akk = tile(A, MT, k, k);

                /* Factorize diagonal tile (highest priority, critcal path)*/
                #pragma omp task depend(inout: Akk[0]) priority(100) \
                                 firstprivate(k, mk, nk, Akk, b, layout)
                {
                    dgetrf_seq(layout, mk, nk, Akk, b);
                }


                /* Update tiles on the same row (U part) */
                for (int j = k + 1; j < NT; ++j) {
                    const int nj = tile_n(N, b, NT, j);
                    double *Akj = tile(A, MT, k, j);

                    #pragma omp task depend(in: Akk[0]) depend(inout: Akj[0]) \
                                     priority(50) \
                                     firstprivate(kk, nj, Akk, Akj, b)
                    {
                        cblas_dtrsm(CblasColMajor,
                                    CblasLeft, CblasLower,
                                    CblasNoTrans, CblasUnit,
                                    kk, nj,
                                    1.0, Akk, b,
                                         Akj, b);
                    }
                }

                /* Update tiles on the same column (L part) */
                for (int i = k + 1; i < MT; ++i) {
                    const int mi = tile_m(M, b, MT, i);
                    double *Aik = tile(A, MT, i, k);

                    #pragma omp task depend(in: Akk[0]) depend(inout: Aik[0]) \
                                     priority(50) \
                                     firstprivate(mi, kk, Akk, Aik, b)
                    {
                        cblas_dtrsm(CblasColMajor,
                                    CblasRight, CblasUpper,
                                    CblasNoTrans, CblasNonUnit,
                                    mi, kk,
                                    1.0, Akk, b,
                                         Aik, b);
                    }
                }
                
                /* Update remaining tiles with GEMM */
                for (int i = k + 1; i < MT; ++i) {
                    const int mi = tile_m(M, b, MT, i);
                    double *Aik = tile(A, MT, i, k);

                    for (int j = k + 1; j < NT; ++j) {
                        const int nj = tile_n(N, b, NT, j);
                        double *Akj = tile(A, MT, k, j);
                        double *Aij = tile(A, MT, i, j);

                        /* Priority based on proximity to diagonal
                         * Tiles closer to diagonal are on critical path for next k */
                        int prio = (i == k+1 || j == k+1) ? 25 : 0;

                        #pragma omp task depend(in: Aik[0], Akj[0]) depend(inout: Aij[0]) \
                                         priority(prio) \
                                         firstprivate(mi, nj, kk, Aik, Akj, Aij, b)
                        {
                            dgemm_seq(CblasColMajor,
                                      CblasNoTrans, CblasNoTrans,
                                      mi, nj, kk,
                                      -1.0, Aik, b,
                                            Akj, b,
                                       1.0, Aij, b);
                        }
                    }
                }

            }
        } 
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_omp __attribute__ ((unused)) = dgetrf_tiled_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_omp;

/**
 * @brief Registration function
 */
void dgetrf_tiled_omp_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_omp_init( void )
{
    fct_dgetrf_tiled_omp.mpi    = 0;
    fct_dgetrf_tiled_omp.tiled  = 1;
    fct_dgetrf_tiled_omp.starpu = 0;
    fct_dgetrf_tiled_omp.name   = "omp-t";
    fct_dgetrf_tiled_omp.helper = "OpenMP tile-based implementation of the dgetrf";
    fct_dgetrf_tiled_omp.fctptr = dgetrf_tiled_omp;
    fct_dgetrf_tiled_omp.next   = NULL;

    register_fct( &fct_dgetrf_tiled_omp, ALGO_GETRF );
}
