      SUBROUTINE DECREASE_PIV(N, PIV)

      INTEGER       N
      INTEGER       PIV( N )

      DO I=1,N
        PIV(I) = PIV(I) - 1
      END DO

      END

      SUBROUTINE WDPSTRF( UPLO, N, A, LDA, PIV, RANK, TOL, WORK, INFO )
*
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   TOL
      INTEGER            INFO, LDA, N, RANK
      CHARACTER          UPLO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), WORK( 2*N )
      INTEGER            PIV( N )
cf2py character optional,intent(in) :: uplo='L'
cf2py integer intent(hide), depend(a) :: n=shape(a,1)
cf2py real*8 intent(in,out,copy,out=factor) :: a
cf2py integer intent(hide),depend(a) :: lda=shape(a,0)
cf2py integer intent(out),depend(n),dimension(n) :: piv
cf2py integer intent(out) :: rank
cf2py real*8 optional,intent(in) :: tol=-1.
cf2py real*8 intent(hide),depend(n),dimension(2*n) :: work
cf2py integer intent(out) :: info
cf2py threadsafe

      EXTERNAL DPSTRF

      CALL DPSTRF( UPLO, N, A, LDA, PIV, RANK, TOL, WORK, INFO )

      CALL DECREASE_PIV( N, PIV )

      RETURN

      END

      SUBROUTINE WSPSTRF( UPLO, N, A, LDA, PIV, RANK, TOL, WORK, INFO )
*
*
*     .. Scalar Arguments ..
      REAL               TOL
      INTEGER            INFO, LDA, N, RANK
      CHARACTER          UPLO
*     ..
*     .. Array Arguments ..
      REAL               A( LDA, * ), WORK( 2*N )
      INTEGER            PIV( N )
cf2py character optional,intent(in) :: uplo='L'
cf2py integer intent(hide), depend(a) :: n=shape(a,1)
cf2py real intent(in,out,copy,out=factor) :: a
cf2py integer intent(hide),depend(a) :: lda=shape(a,0)
cf2py integer intent(out),depend(n),dimension(n) :: piv
cf2py integer intent(out) :: rank
cf2py real optional,intent(in) :: tol=-1.
cf2py real intent(hide),depend(n),dimension(2*n) :: work
cf2py integer intent(out) :: info
cf2py threadsafe

      EXTERNAL SPSTRF

      CALL SPSTRF( UPLO, N, A, LDA, PIV, RANK, TOL, WORK, INFO )

      CALL DECREASE_PIV( N, PIV )

      RETURN

      END