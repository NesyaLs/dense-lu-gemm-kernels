/**
 * @file dgetrf_omp2.c
 *
 * @brief OpenMP "dedicated" LU factorization (no pivoting) with a persistent
 *        parallel region (reduces fork-join overhead).
 *
 * Strategy:
 *  - Panel factorization + TRSM are executed by a single thread (dependencies).
 *  - Trailing submatrix update (A22) is parallelized by distributing blocks
 *    across threads with omp for.
 *  - Uses your own sequential DGEMM (dgemm_seq) for the update.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 */
#include "myblas.h"
#include <assert.h>
#include <omp.h>
#include <string.h>

int dgetrf_nbloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda);

int dgemm_seq( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc );

int
dgetrf_omp2( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    assert(layout == CblasColMajor);
    assert(lda >= M);

    const int K = (M < N) ? M : N;

    const int b = 64;

    const int ib  = 128;
    const int jb2 = 128;

    #pragma omp parallel
    {
        for (int j = 0; j < K; j += b) {

            const int jb = ((j + b) < K) ? b : (K - j);
            #pragma omp single
            {
                // Panel factorization (seq)
                dgetrf_nbloc(layout, M - j, jb, &A[j + j*lda], lda);

                // TRSM to compute U12
                if (j + jb < N) {
                    cblas_dtrsm(CblasColMajor,
                                CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                                jb, N - (j + jb),
                                1.0,
                                &A[j + j*lda], lda,
                                &A[j + (j+jb)*lda], lda);
                }
            }

            // ensure all threads have updated panel 
            #pragma omp barrier

            if ((j + jb < M) && (j + jb < N)) {

                const int Mm = M - (j + jb);
                const int Nn = N - (j + jb);

                #pragma omp for collapse(2) schedule(guided)
                for (int jj = 0; jj < Nn; jj += jb2) {
                    for (int ii = 0; ii < Mm; ii += ib) {

                        const int curM = (ii + ib  < Mm) ? ib  : (Mm - ii);
                        const int curN = (jj + jb2 < Nn) ? jb2 : (Nn - jj);

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

            #pragma omp barrier
        }
    }

    return ALGONUM_SUCCESS;
}

/* Registration */
static dgetrf_fct_t valid_dgetrf_omp2 __attribute__ ((unused)) = dgetrf_omp2;

fct_list_t fct_dgetrf_omp2;

void dgetrf_omp2_init( void ) __attribute__( ( constructor ) );
void
dgetrf_omp2_init( void )
{
    memset(&fct_dgetrf_omp2, 0, sizeof(fct_list_t));
    fct_dgetrf_omp2.mpi    = 0;
    fct_dgetrf_omp2.tiled  = 0;
    fct_dgetrf_omp2.starpu = 0;
    fct_dgetrf_omp2.name   = "omp2";
    fct_dgetrf_omp2.helper = "OpenMP dedicated LU (persistent parallel region + dgemm_seq update)";
    fct_dgetrf_omp2.fctptr = dgetrf_omp2;
    fct_dgetrf_omp2.next   = NULL;

    register_fct( &fct_dgetrf_omp2, ALGO_GETRF );
}