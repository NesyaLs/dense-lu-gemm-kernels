/**
 * @file dgetrf_nbloc.c
 *
 * @brief Unblocked LU factorization (no pivoting).
 *
 * Implémente l'algorithme classique de Gauss (LU sans pivot),
 * en place, sur une matrice MxN en stockage colonne-major.
 *
 * Le résultat :
 *  - partie strictement triangulaire inférieure = L (diag implicite = 1)
 *  - partie triangulaire supérieure (diag incluse) = U
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 */
#include "myblas.h"
#include <string.h>
#include <stdio.h>

int dgetrf_nbloc(CBLAS_LAYOUT layout, int M, int N, double *A, int lda)
{
    if (layout != CblasColMajor) {
        fprintf(stderr, "dgetrf_nbloc: only CblasColMajor is valid\n");
        return ALGONUM_NOT_IMPLEMENTED;
    }

    int K = (M < N) ? M : N;

    for (int k = 0; k < K; k++) {
        double piv = A[k + k * lda];

        for (int i = k + 1; i < M; i++) {
            A[i + k * lda] /= piv;  // L(i,k)
        }

        for (int j = k + 1; j < N; j++) {
            double Ukj = A[k + j * lda];  // U(k,j)
            for (int i = k + 1; i < M; i++) {
                A[i + j * lda] -= A[i + k * lda] * Ukj;
            }
        }
    }

    return ALGONUM_SUCCESS;
}

static dgetrf_fct_t valid_dgetrf_nbloc __attribute__((unused)) = dgetrf_nbloc;

fct_list_t fct_dgetrf_nbloc;

void dgetrf_nbloc_init(void) __attribute__((constructor));
void dgetrf_nbloc_init(void)
{
    memset(&fct_dgetrf_nbloc, 0, sizeof(fct_list_t));
    fct_dgetrf_nbloc.mpi    = 0;
    fct_dgetrf_nbloc.tiled  = 0;
    fct_dgetrf_nbloc.starpu = 0;
    fct_dgetrf_nbloc.name   = "nbloc";
    fct_dgetrf_nbloc.helper = "Unblocked LU (naive Gauss, no pivot)";
    fct_dgetrf_nbloc.fctptr = dgetrf_nbloc;
    fct_dgetrf_nbloc.next   = NULL;

    register_fct(&fct_dgetrf_nbloc, ALGO_GETRF);
}
