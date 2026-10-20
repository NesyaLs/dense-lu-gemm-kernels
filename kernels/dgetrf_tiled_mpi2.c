/**
 *
 * @file dgetrf_tiled_mpi2.c
 *
 * @brief MPI tile-based dgetrf with non-blocking communications.
 *        Derived from the reference MPI tile-based dgetrf of mini-chameleon.
 *
 * @authors Arthur Dufetrelle (team project)
 * @date    2025-2026
 *
 */
#include "myblas.h"
#include <string.h>

int
dgetrf_tiled_mpi2( CBLAS_LAYOUT layout,
                   int M, int N, int b, double **A )
{
#if !defined(ENABLE_MPI)
    (void)layout; (void)M; (void)N; (void)b; (void)A;
    return ALGONUM_NOT_IMPLEMENTED;
#else
    MPI_Status status;
    double Akk[b*b], Amk[b*b];
    double *Akkptr, *Amkptr;
    int myrank = global_options.mpirank;

    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_imin( MT, NT );

    /* Buffers pour recevoir les U[k,n] - un par colonne */
    double *Akn_buffers = (double *)malloc( NT * b * b * sizeof(double) );
    
    /* Tableaux de requests pour les communications non-bloquantes */
    MPI_Request *send_requests = (MPI_Request *)malloc( NT * MT * sizeof(MPI_Request) );
    MPI_Request *recv_requests = (MPI_Request *)malloc( NT * sizeof(MPI_Request) );
    int *recv_pending = (int *)calloc( NT, sizeof(int) );  /* 1 si réception en cours */
    int num_sends;

    for ( int k = 0; k < KT; k++ ) {
        int mk = ( k == (MT-1) ) ? M - k * b : b;
        int nk = ( k == (NT-1) ) ? N - k * b : b;
        int kk = my_imin( mk, nk );
        num_sends = 0;

        /*
         * =====================================================
         * Étape 1: Factorisation diagonale + broadcast
         * =====================================================
         */
        if ( get_rank_of( k, k ) == myrank ) {
            Akkptr = A[ MT * k + k ];
            dgetrf_seq( LAPACK_COL_MAJOR, mk, nk, Akkptr, b );
        }
        else {
            Akkptr = Akk;
        }
        MPI_Bcast( Akkptr, b*b, MPI_DOUBLE, get_rank_of( k, k ), MPI_COMM_WORLD );

        /*
         * =====================================================
         * Étape 2: TRSM sur la ligne k
         * =====================================================
         */
        for ( int n = k+1; n < NT; n++ ) {
            int nn = ( n == (NT-1) ) ? N - n * b : b;
            if ( get_rank_of( k, n ) == myrank ) {
                cblas_dtrsm( CblasColMajor, CblasLeft, CblasLower,
                             CblasNoTrans, CblasUnit,
                             kk, nn,
                             1., Akkptr,          b,
                                 A[ MT * n + k ], b );
            }
        }

        /*
         * =====================================================
         * Étape 3: TRSM sur la colonne k + broadcast
         * =====================================================
         */
        for ( int m = k+1; m < MT; m++ ) {
            int mm = ( m == (MT-1) ) ? M - m * b : b;
            if ( get_rank_of( m, k ) == myrank ) {
                Amkptr = A[ MT * k + m ];
                cblas_dtrsm( CblasColMajor, CblasRight, CblasUpper,
                             CblasNoTrans, CblasNonUnit,
                             mm, kk,
                             1., Akkptr, b,
                                 Amkptr, b );
            }
        }

        /*
         * =====================================================
         * Étape 4: Lancer TOUTES les communications U[k,n]
         *          AVANT de faire les GEMM
         * =====================================================
         */
        
        /* 4a: Lancer les réceptions non-bloquantes */
        for ( int n = k+1; n < NT; n++ ) {
            recv_pending[n] = 0;
            
            /* Est-ce que je possède au moins une tuile dans la colonne n sous la diagonale ? */
            int need_Ukn = 0;
            for ( int m = k+1; m < MT && !need_Ukn; m++ ) {
                if ( get_rank_of( m, n ) == myrank && get_rank_of( k, n ) != myrank ) {
                    need_Ukn = 1;
                }
            }
            
            if ( need_Ukn ) {
                double *Akn_buf = Akn_buffers + n * b * b;
                MPI_Irecv( Akn_buf, b*b, MPI_DOUBLE,
                           get_rank_of( k, n ), k * NT + n,
                           MPI_COMM_WORLD, &recv_requests[n] );
                recv_pending[n] = 1;
            }
        }

        /* 4b: Lancer les envois non-bloquants */
        for ( int n = k+1; n < NT; n++ ) {
            if ( get_rank_of( k, n ) == myrank ) {
                double *Aknptr = A[ MT * n + k ];
                
                /* Envoyer à tous ceux qui en ont besoin */
                for ( int m = k+1; m < MT; m++ ) {
                    int dest = get_rank_of( m, n );
                    if ( dest != myrank ) {
                        MPI_Isend( Aknptr, b*b, MPI_DOUBLE,
                                   dest, k * NT + n,
                                   MPI_COMM_WORLD, &send_requests[num_sends++] );
                    }
                }
            }
        }

        /*
         * =====================================================
         * Étape 5: GEMM - d'abord les locaux, puis ceux qui
         *          nécessitent des données distantes
         * =====================================================
         */
        
        /* 5a: GEMM pour les tuiles où je possède AUSSI U[k,n] (pas de comm) */
        for ( int m = k+1; m < MT; m++ ) {
            int mm = ( m == (MT-1) ) ? M - m * b : b;
            
            if ( get_rank_of( m, k ) == myrank ) {
                Amkptr = A[ MT * k + m ];
            }
            else {
                Amkptr = Amk;
            }
            /* Broadcast L[m,k] - nécessaire pour tous */
            MPI_Bcast( Amkptr, b*b, MPI_DOUBLE, get_rank_of( m, k ), MPI_COMM_WORLD );

            for ( int n = k+1; n < NT; n++ ) {
                int nn = ( n == (NT-1) ) ? N - n * b : b;

                if ( get_rank_of( m, n ) == myrank ) {
                    double *Aknptr;
                    
                    if ( get_rank_of( k, n ) == myrank ) {
                        /* Données locales - pas d'attente */
                        Aknptr = A[ MT * n + k ];
                    }
                    else {
                        /* Attendre la réception de U[k,n] */
                        if ( recv_pending[n] ) {
                            MPI_Wait( &recv_requests[n], &status );
                            recv_pending[n] = 0;  /* Ne plus attendre pour les prochains m */
                        }
                        Aknptr = Akn_buffers + n * b * b;
                    }

                    dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                               mm, nn, kk,
                               -1., Amkptr, b,
                                    Aknptr, b,
                               1.,  A[ MT * n + m ], b );
                }
            }
        }

        /* Attendre que tous les envois soient terminés */
        if ( num_sends > 0 ) {
            MPI_Waitall( num_sends, send_requests, MPI_STATUSES_IGNORE );
        }
    }

    free( Akn_buffers );
    free( send_requests );
    free( recv_requests );
    free( recv_pending );

    return ALGONUM_SUCCESS;
#endif
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_mpi2 __attribute__ ((unused)) = dgetrf_tiled_mpi2;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_mpi2;

/**
 * @brief Registration function
 */
void dgetrf_tiled_mpi2_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_mpi2_init( void )
{
    fct_dgetrf_tiled_mpi2.mpi    = 1;
    fct_dgetrf_tiled_mpi2.tiled  = 1;
    fct_dgetrf_tiled_mpi2.starpu = 0;
    fct_dgetrf_tiled_mpi2.name   = "mpi2";
    fct_dgetrf_tiled_mpi2.helper = "MPI tile-based implementation of GETRF (mpi2)";
    fct_dgetrf_tiled_mpi2.fctptr = dgetrf_tiled_mpi2;
    fct_dgetrf_tiled_mpi2.next   = NULL;

    register_fct( &fct_dgetrf_tiled_mpi2, ALGO_GETRF );
}