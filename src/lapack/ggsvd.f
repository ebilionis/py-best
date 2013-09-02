      SUBROUTINE WDGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
     $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ,
     $                   LWORK, WORK, IWORK, INFO )
*
*     .. Scalar Arguments ..
      CHARACTER          JOBQ, JOBU, JOBV
      INTEGER            INFO, K, L, LDA, LDB, LDQ, LDU, LDV, M, N, P,
     $                   LWORK
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( N )
      DOUBLE PRECISION   A( M, N ), ALPHA( N ), B( P, N ),
     $                   BETA( N ), Q( N, N ), U( M, M ),
     $                   V( P, P ), WORK( LWORK )
cf2py character optional,intent(in) :: jobq='Q'
cf2py character optional,intent(in) :: jobu='U'
cf2py character optional,intent(in) :: jobv='N'
cf2py integer intent(hide),depend(a) :: m=shape(a,0)
cf2py integer intent(hide),depend(a) :: n=shape(a,1)
cf2py integer intent(hide),depend(b) :: p=shape(b,0)
cf2py integer intent(out) :: k
cf2py integer intent(out) :: l
cf2py real*8 intent(in,out,copy),dimension(m,n) :: a
cf2py integer intent(hide),depend(m,a) :: lda=m
cf2py real*8 intent(in,out,copy),dimension(p,n) :: b
cf2py integer intent(hide),depend(p,b) :: ldb=p
cf2py real*8 intent(out),dimension(n),depend(n) :: alpha
cf2py real*8 intent(out),dimension(n),depend(n) :: beta
cf2py real*8 intent(out),dimension(m,m),depend(m) :: u
cf2py integer intent(hide),depend(m,u) :: ldu=m
cf2py real*8 intent(out),depend(p),dimension(p,p),depend(p) :: v
cf2py integer intent(hide),depend(p,v) :: ldv=p
cf2py real*8 intent(out),depend(n),dimension(n,n),depend(n) :: q
cf2py integer intent(hide),depend(n,q) :: ldq=n
cf2py integer intent(hide),depend(m,n,p) :: lwork=max(max(4*n,m+n),p+n)
cf2py real*8 optional,intent(hide),dimension(lwork),depend(lwork) :: work
cf2py integer intent(hide,out,out=sort),depend(n),dimension(n) :: iwork
cf2py integer intent(out) :: info
cf2py threadsafe

      EXTERNAL DGGSVD

      CALL       DGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
     $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ, WORK,
     $                   IWORK, INFO )

      RETURN

      END

      SUBROUTINE WSGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
     $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ,
     $                   LWORK, WORK, IWORK, INFO )
*
*     .. Scalar Arguments ..
      CHARACTER          JOBQ, JOBU, JOBV
      INTEGER            INFO, K, L, LDA, LDB, LDQ, LDU, LDV, M, N, P,
     $                   LWORK
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( N )
      REAL               A( M, N ), ALPHA( N ), B( P, N ),
     $                   BETA( N ), Q( N, N ), U( M, M ),
     $                   V( P, P ), WORK( LWORK )
cf2py character optional,intent(in) :: jobq='Q'
cf2py character optional,intent(in) :: jobu='U'
cf2py character optional,intent(in) :: jobv='N'
cf2py integer intent(hide),depend(a) :: m=shape(a,0)
cf2py integer intent(hide),depend(a) :: n=shape(a,1)
cf2py integer intent(hide),depend(b) :: p=shape(b,0)
cf2py integer intent(out) :: k
cf2py integer intent(out) :: l
cf2py real intent(in,out,copy),dimension(m,n) :: a
cf2py integer intent(hide),depend(m,a) :: lda=m
cf2py real intent(in,out,copy),dimension(p,n) :: b
cf2py integer intent(hide),depend(p,b) :: ldb=p
cf2py real intent(out),dimension(n),depend(n) :: alpha
cf2py real intent(out),dimension(n),depend(n) :: beta
cf2py real intent(out),dimension(m,m),depend(m) :: u
cf2py integer intent(hide),depend(m,u) :: ldu=m
cf2py real intent(out),depend(p),dimension(p,p),depend(p) :: v
cf2py integer intent(hide),depend(p,v) :: ldv=p
cf2py real intent(out),depend(n),dimension(n,n),depend(n) :: q
cf2py integer intent(hide),depend(n,q) :: ldq=n
cf2py integer intent(hide),depend(m,n,p) :: lwork=max(max(4*n,m+n),p+n)
cf2py real optional,intent(hide),dimension(lwork),depend(lwork) :: work
cf2py integer intent(hide,out,out=sort),depend(n),dimension(n) :: iwork
cf2py integer intent(out) :: info
cf2py threadsafe

      EXTERNAL SGGSVD

      CALL       SGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
     $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ, WORK,
     $                   IWORK, INFO )

      RETURN

      END