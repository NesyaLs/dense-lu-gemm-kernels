/**
 *
 * @file dgetrf_tiled_mpi_omp.c
 *
 * @brief Hybrid MPI+OpenMP tile-based implementation of the dgetrf.
 *        Derived from the reference MPI tile-based dgetrf of mini-chameleon.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"

/*
 * Fonctions utilitaires
 */
static inline int ceil_div(int x, int y) { return (x + y - 1) / y; }
static inline int min_int(int a, int b)  { return (a < b) ? a : b; }

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
dgetrf_tiled_mpi_omp( CBLAS_LAYOUT layout,
                      int M, int N, int b, double **A )
{
#if !defined(ENABLE_MPI)
    (void)layout; (void)M; (void)N; (void)b; (void)A;
    return ALGONUM_NOT_IMPLEMENTED;
#else
    assert(layout == CblasColMajor);

    MPI_Status status;
    MPI_Request request;
    double Akk[b*b], Amk[b*b], Akn[b*b];
    double *Akkptr, *Amkptr, *Aknptr;
    int myrank = global_options.mpirank;

    const int MT = ceil_div(M, b);
    const int NT = ceil_div(N, b);
    const int KT = min_int(MT, NT);
    int m, n, k;

    for ( k = 0; k < KT; k++ ) {
        int mk = tile_m(M, b, MT, k);
        int nk = tile_n(N, b, NT, k);
        int kk = min_int(mk, nk);

        if ( get_rank_of( k, k ) == myrank ) {
            Akkptr = tile(A, MT, k, k);
            dgetrf_seq( LAPACK_COL_MAJOR, mk, nk, Akkptr, b );
        }
        else {
            Akkptr = Akk;
        }
        MPI_Bcast( Akkptr, b*b, MPI_DOUBLE, get_rank_of( k, k ), MPI_COMM_WORLD );

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                /* TRSM sur la ligne k (U part) - tuiles locales */
                for ( n = k+1; n < NT; n++ ) {
                    if ( get_rank_of( k, n ) == myrank ) {
                        int nn = tile_n(N, b, NT, n);
                        double *Akn_local = tile(A, MT, k, n);

                        #pragma omp task firstprivate(kk, nn, Akkptr, Akn_local, b) \
                                         priority(50)
                        {
                            cblas_dtrsm( CblasColMajor, CblasLeft, CblasLower,
                                         CblasNoTrans, CblasUnit,
                                         kk, nn,
                                         1., Akkptr,    b,
                                             Akn_local, b );
                        }
                    }
                }

                /* TRSM sur la colonne k (L part) - tuiles locales */
                for ( m = k+1; m < MT; m++ ) {
                    if ( get_rank_of( m, k ) == myrank ) {
                        int mm = tile_m(M, b, MT, m);
                        double *Amk_local = tile(A, MT, m, k);

                        #pragma omp task firstprivate(mm, kk, Akkptr, Amk_local, b) \
                                         priority(50)
                        {
                            cblas_dtrsm( CblasColMajor, CblasRight, CblasUpper,
                                         CblasNoTrans, CblasNonUnit,
                                         mm, kk,
                                         1., Akkptr,    b,
                                             Amk_local, b );
                        }
                    }
                }
            }
        }
        for ( m = k+1; m < MT; m++ ) {
            int mm = tile_m(M, b, MT, m);

            /* Broadcast de L[m,k] */
            if ( get_rank_of( m, k ) == myrank ) {
                Amkptr = tile(A, MT, m, k);
            }
            else {
                Amkptr = Amk;
            }
            MPI_Bcast( Amkptr, b*b, MPI_DOUBLE, get_rank_of( m, k ), MPI_COMM_WORLD );

            /*
             * GEMM avec parallélisme OpenMP pour les tuiles locales
             */
            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    for ( n = k+1; n < NT; n++ ) {
                        int nn = tile_n(N, b, NT, n);

                        if ( get_rank_of( m, n ) == myrank ) {
                            /* Je possède A[m,n] */
                            double *Aknptr_local;
                            double *Amn_local = tile(A, MT, m, n);

                            if ( get_rank_of( k, n ) == myrank ) {
                                /* Je possède aussi U[k,n] */
                                Aknptr_local = tile(A, MT, k, n);

                                /* Priorité: tuiles proches de la diagonale */
                                int prio = (m == k+1 || n == k+1) ? 25 : 0;

                                #pragma omp task firstprivate(mm, nn, kk, Amkptr, Aknptr_local, Amn_local, b) \
                                                 priority(prio)
                                {
                                    dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                                               mm, nn, kk,
                                               -1., Amkptr,       b,
                                                    Aknptr_local, b,
                                               1.,  Amn_local,    b );
                                }
                            }
                            else {
                                /* Je dois recevoir U[k,n] - fait en dehors du parallel */
                            }
                        }
                    }
                }
            }

            for ( n = k+1; n < NT; n++ ) {
                int nn = tile_n(N, b, NT, n);

                if ( get_rank_of( m, n ) == myrank ) {
                    if ( get_rank_of( k, n ) != myrank ) {
                        /* Je dois recevoir U[k,n] puis faire le GEMM */
                        Aknptr = Akn;
                        MPI_Recv( Aknptr, b*b, MPI_DOUBLE,
                                  get_rank_of( k, n ), MT * n + k,
                                  MPI_COMM_WORLD, &status );

                        double *Amn_local = tile(A, MT, m, n);
                        dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                                   mm, nn, kk,
                                   -1., Amkptr,    b,
                                        Aknptr,    b,
                                   1.,  Amn_local, b );
                    }
                }
                else {
                    /* Envoi non-bloquant si je possède U[k,n] */
                    if ( get_rank_of( k, n ) == myrank ) {
                        Aknptr = tile(A, MT, k, n);
                        MPI_Isend( Aknptr, b*b, MPI_DOUBLE,
                                   get_rank_of( m, n ), MT * n + k,
                                   MPI_COMM_WORLD, &request );
                    }
                }
            }
        }

        /* Synchronisation avant l'itération suivante */
        MPI_Barrier( MPI_COMM_WORLD );
    }

    return ALGONUM_SUCCESS;
#endif
}

/* Vérification du prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_mpi_omp __attribute__ ((unused)) = dgetrf_tiled_mpi_omp;

/* Variable d'enregistrement */
fct_list_t fct_dgetrf_tiled_mpi_omp;

/**
 * @brief Fonction d'enregistrement
 */
void dgetrf_tiled_mpi_omp_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_mpi_omp_init( void )
{
    fct_dgetrf_tiled_mpi_omp.mpi    = 1;
    fct_dgetrf_tiled_mpi_omp.tiled  = 1;
    fct_dgetrf_tiled_mpi_omp.starpu = 0;
    fct_dgetrf_tiled_mpi_omp.name   = "mpi_omp";
    fct_dgetrf_tiled_mpi_omp.helper = "Hybrid MPI+OpenMP tile-based dgetrf";
    fct_dgetrf_tiled_mpi_omp.fctptr = dgetrf_tiled_mpi_omp;
    fct_dgetrf_tiled_mpi_omp.next   = NULL;

    register_fct( &fct_dgetrf_tiled_mpi_omp, ALGO_GETRF );
}