/**
 *
 * @file dgetrf_omp.c
 *
 * @brief OpenMP (fork-join) implementation of the dgetrf.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"
#include <assert.h>

int dgetrf_nbloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda);

int dgetrf_omp( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    assert(layout == CblasColMajor);
    assert(lda >= M);

    int j, jb;
    int K = (M < N) ? M : N;

    const int b   = 64;
    const int ib  = 128;
    const int jb2 = 128;

    for (j = 0; j < K; j += b) {
        jb = ((j + b) < K) ? b : (K - j);

        dgetrf_nbloc(layout, M - j, jb, &A[j + j*lda], lda);

        if (j + jb < N) {
            cblas_dtrsm(CblasColMajor,
                        CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        jb, N - (j + jb),
                        1.0,
                        &A[j + j*lda], lda,
                        &A[j + (j+jb)*lda], lda);
        }

        if ((j + jb < M) && (j + jb < N)) {
            int Mm = M - (j + jb);
            int Nn = N - (j + jb);

            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int jj = 0; jj < Nn; jj += jb2) {
                for (int ii = 0; ii < Mm; ii += ib) {

                    int curM = (ii + ib  < Mm) ? ib  : (Mm - ii);
                    int curN = (jj + jb2 < Nn) ? jb2 : (Nn - jj);

                    dgemm_seq(CblasColMajor, CblasNoTrans, CblasNoTrans,
                              curM, curN, jb,
                              -1.0,
                              &A[(j+jb+ii) + (j      )*lda], lda,
                              &A[(j      ) + (j+jb+jj)*lda], lda,
                               1.0,
                              &A[(j+jb+ii) + (j+jb+jj)*lda], lda);
                }
            }
        }
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_omp __attribute__ ((unused)) = dgetrf_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_omp;

/**
 * @brief Registration function
 */
void dgetrf_omp_init( void ) __attribute__( ( constructor ) );
void
dgetrf_omp_init( void )
{
    fct_dgetrf_omp.mpi    = 0;
    fct_dgetrf_omp.tiled  = 0;
    fct_dgetrf_omp.starpu = 0;
    fct_dgetrf_omp.name   = "omp";
    fct_dgetrf_omp.helper = "OpenMP implementation of the dgetrf";
    fct_dgetrf_omp.fctptr = dgetrf_omp;
    fct_dgetrf_omp.next   = NULL;

    register_fct( &fct_dgetrf_omp, ALGO_GETRF );
}
