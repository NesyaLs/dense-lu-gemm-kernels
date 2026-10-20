/**
 *
 * @file dgetrf_seq.c
 *
 * @brief Sequential dgetrf (vectorized AVX2 kernel, blocked entry point).
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"
#include <assert.h>

#include <immintrin.h>

int dgetrf_bloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda);

int
dgetrf_seq_vect( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int k, i, j;
    int K = (M < N) ? M : N;

    for( k = 0; k < K; k++ ) {
        double pivot = A[k + k * lda];
        
        if( pivot == 0.0 ) {
            return ALGONUM_FAIL;
        }

        for( i = k + 1; i < M; i++ ) {
            A[i + k * lda] /= pivot;
        }

        for( j = k + 1; j < N; j++ ) {
            __m256d akj_vec = _mm256_set1_pd( A[k + j * lda] );
            
            i = k + 1;
            for( ; i + 4 <= M; i += 4 ) {
                __m256d aik_vec = _mm256_loadu_pd( &A[i + k * lda] );
                __m256d aij_vec = _mm256_loadu_pd( &A[i + j * lda] );
                __m256d result = _mm256_sub_pd( aij_vec, _mm256_mul_pd( aik_vec, akj_vec ) );
                _mm256_storeu_pd( &A[i + j * lda], result );
            }
            for( ; i < M; i++ ) {
                A[i + j * lda] -= A[i + k * lda] * A[k + j * lda];
            }
        }
    }

    return ALGONUM_SUCCESS;
}

int dgetrf_seq( CBLAS_LAYOUT layout, int M, int N, double *A, int lda ) {
    return dgetrf_bloc( layout, M, N, A, lda );
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_seq __attribute__ ((unused)) = dgetrf_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_seq;

/**
 * @brief Registration function
 */
void dgetrf_seq_init( void ) __attribute__( ( constructor ) );
void
dgetrf_seq_init( void )
{
    fct_dgetrf_seq.mpi    = 0;
    fct_dgetrf_seq.tiled  = 0;
    fct_dgetrf_seq.starpu = 0;
    fct_dgetrf_seq.name   = "seq";
    fct_dgetrf_seq.helper = "Basic sequential implementation of the dgetrf";
    fct_dgetrf_seq.fctptr = dgetrf_seq;
    fct_dgetrf_seq.next   = NULL;

    register_fct( &fct_dgetrf_seq, ALGO_GETRF );
}
