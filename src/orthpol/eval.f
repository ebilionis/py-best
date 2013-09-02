      SUBROUTINE SEVAL(P, ALPHA, BETA, GAMMA, X, PHI)

      INTEGER   P
      REAL      X
      REAL      ALPHA( P ), BETA( P ), GAMMA( P ), PHI( P+1 )
cf2py integer intent(hide),depend(alpha) :: p=len(alpha)
cf2py real intent(in) :: alpha
cf2py real intent(in),depend(p),dimension(p) :: beta
cf2py real intent(in),depend(p),dimension(p) :: gamma
cf2py real intent(in) :: x
cf2py real intent(out),depend(p),dimension(p+1) :: phi
      PHI(:) = 0.
      PHI(1) = 1. / GAMMA(1)
      IF (P.GE.1) THEN
        PHI(2) = (X - ALPHA(1)) * (PHI(1) / GAMMA(1))
      END IF
      DO I=3,P+1
        PHI(I) = ((X - ALPHA(I-1)) * PHI(I-1) - BETA(I-1) * PHI(I-2)) /
     $           GAMMA(I)
      END DO

      RETURN

      END SUBROUTINE

      SUBROUTINE SEVAL_ALL(N, P, ALPHA, BETA, GAMMA, X, PHI)

      INTEGER   N, P
      REAL      ALPHA( P ), BETA( P ), GAMMA( P ), X( N ),
     $          PHI( N, P + 1)
cf2py integer intent(hide),depend(x) :: n=len(x)
cf2py integer intent(hide),depend(alpha) :: p=len(alpha)
cf2py real intent(in) :: alpha
cf2py real intent(in),depend(p),dimension(p) :: beta
cf2py real intent(in),depend(p),dimension(p) :: gamma
cf2py real intent(in) :: x
cf2py real intent(out),depend(n,p),dimension(n,p) :: phi
      PHI(:,:) = 0.
      PHI(:,1) = 1. / GAMMA(1)
      IF (P.GE.1) THEN
        PHI(:,2) = (X - ALPHA(1)) * (PHI(:,1) / GAMMA(1))
      END IF
      DO I=3,P+1
        PHI(:,I) = ((X - ALPHA(I-1)) * PHI(:,I-1)
     $              - BETA(I-1) * PHI(:,I-2)) / GAMMA(I)
      END DO

      RETURN

      END SUBROUTINE