/**
 *
 * @file dgemm_tiled.c
 *
 * @brief Sequential tiled DGEMM (tile layout + AVX2 micro-kernel).
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"
#include <immintrin.h>

static int dgemm_tiled_block_size = -1;

#define BLOCK_SIZE 120

void free_tile_array(int M, int N, int b, double **Atile) {
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    
    for(int i = 0; i < MT * NT; i++) {
        free(Atile[i]);
    }
    free(Atile);
}

int dgemm_tiled(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                CBLAS_TRANSPOSE transB, const int M, const int N,
                const int K, const double alpha, const double *A,
                const int lda, const double *B, const int ldb,
                const double beta, double *C, const int ldc)
{
    if (layout != CblasColMajor || transA != CblasNoTrans || transB != CblasNoTrans)
        return ALGONUM_NOT_IMPLEMENTED;

    int b = BLOCK_SIZE;
    
    int MT = (M + b - 1) / b;
    int NT = (N + b - 1) / b;
    int KT = (K + b - 1) / b;

    int m, n, k;
    int mm, nn, kk;

    double **Atile = lapack2tile(M, K, b, A, lda);
    double **Btile = lapack2tile(K, N, b, B, ldb);
    double **Ctile = lapack2tile(M, N, b, C, ldc);

    for (n = 0; n < NT; n++) {
        nn = (n == NT - 1) ? N - n * b : b;

        for (m = 0; m < MT; m++) {
            mm = (m == MT - 1) ? M - m * b : b;
            
            for (k = 0; k < KT; k++) {
                kk = (k == KT - 1) ? K - k * b : b;
                double lbeta = (k == 0) ? beta : 1.0;

                process_tile(layout, transA, transB,
                             mm, nn, kk,
                             alpha, 
                             Atile[MT * k + m], b,
                             Btile[KT * n + k], b,
                             lbeta,
                             Ctile[MT * n + m], b);
            }
        }
    }

    tile2lapack(M, N, b, (const double**)Ctile, C, ldc);

    free_tile_array(M, K, b, Atile);
    free_tile_array(K, N, b, Btile);
    free_tile_array(M, N, b, Ctile);

    return ALGONUM_SUCCESS;
}


int dgemm_blocked(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                      CBLAS_TRANSPOSE transB, const int M, const int N,
                      const int K, const double alpha, const double *A,
                      const int lda, const double *B, const int ldb,
                      const double beta, double *C, const int ldc)
{
    int m, n, k, mm, nn, kk;
    double lbeta;
    
    for (mm = 0; mm < M; mm += BLOCK_SIZE) {
        int Mb = (mm + BLOCK_SIZE < M) ? BLOCK_SIZE : M-mm;
        for (nn = 0; nn < N; nn += BLOCK_SIZE) {
            int Nb = (nn + BLOCK_SIZE < N) ? BLOCK_SIZE : N-nn;
            for (kk = 0; kk < K; kk += BLOCK_SIZE) {
                int Kb = (kk + BLOCK_SIZE < K) ? BLOCK_SIZE : K-kk;
                lbeta = (kk == 0) ? beta : 1.;
                process_tile(layout, transA, transB, 
                                    Mb, Nb, Kb, alpha, 
                                    &A[lda*kk + mm], lda,
                                    &B[ldb*nn + kk], ldb, lbeta,
                                    &C[ldc*nn + mm], ldc);
            }
        }
    }
    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_tiled __attribute__ ((unused)) = dgemm_tiled;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_tiled;

/**
 * @brief Registration function
 */
void dgemm_tiled_init( void ) __attribute__( ( constructor ) );
void
dgemm_tiled_init( void )
{
    fct_dgemm_tiled.mpi    = 0;
    fct_dgemm_tiled.tiled  = 0;
    fct_dgemm_tiled.starpu = 0;
    fct_dgemm_tiled.name   = "til";
    fct_dgemm_tiled.helper = "Sequential tiled version of DGEMM";
    fct_dgemm_tiled.fctptr = dgemm_tiled;
    fct_dgemm_tiled.next   = NULL;

    register_fct( &fct_dgemm_tiled, ALGO_GEMM );

    /* Read the value of dgemm_block_size */
    dgemm_tiled_block_size = myblas_getenv_value_int( "BLOCKSIZE", 32 );
}
