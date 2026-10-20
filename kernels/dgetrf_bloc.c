/**
 *
 * @file dgetrf_bloc.c
 *
 * @brief Blocked implementation of the dgetrf.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"
#include <assert.h>

int dgetrf_nbloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda);
int dgetrf_seq_vect(CBLAS_LAYOUT layout, int M, int N, double *A, int lda);

int dgetrf_bloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda)
{
    int j, jb;
    int K = (M < N) ? M : N;
    const int b = 64;

    assert(layout == CblasColMajor);
    assert(lda >= M);

    for (j = 0; j < K; j += b) {
        jb = ((j + b) < K) ? b : (K - j);

        /* 1) Panel factorization */
        dgetrf_seq_vect(layout, M - j, jb, &A[j + j*lda], lda);

        /* 2) TRSM: compute U12 */
        if (j + jb < N) {
            cblas_dtrsm(CblasColMajor,
                        CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        jb, N - (j + jb),
                        1.0,
                        &A[j + j*lda], lda,
                        &A[j + (j+jb)*lda], lda);

            /* 3) Update A22 = A22 - L21 * U12 */
            if (j + jb < M) {
                dgemm_seq(CblasColMajor,
                          CblasNoTrans, CblasNoTrans,
                          M - (j + jb), N - (j + jb), jb,
                          -1.0,
                          &A[(j+jb) +  j     *lda], lda,  /* L21 */
                          &A[ j      + (j+jb)*lda], lda,  /* U12 */
                           1.0,
                          &A[(j+jb) + (j+jb)*lda], lda);  /* A22 */
            }
        }
    }

    return ALGONUM_SUCCESS;
}
/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_bloc __attribute__ ((unused)) = dgetrf_bloc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_bloc;

/**
 * @brief Registration function
 */
void dgetrf_bloc_init( void ) __attribute__( ( constructor ) );
void
dgetrf_bloc_init( void )
{
    fct_dgetrf_bloc.mpi    = 0;
    fct_dgetrf_bloc.tiled  = 0;
    fct_dgetrf_bloc.starpu = 0;
    fct_dgetrf_bloc.name   = "bloc";
    fct_dgetrf_bloc.helper = "Blocked implementation of the dgetrf";
    fct_dgetrf_bloc.fctptr = dgetrf_bloc;
    fct_dgetrf_bloc.next   = NULL;

    register_fct( &fct_dgetrf_bloc, ALGO_GETRF );
}
