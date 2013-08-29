#include <boost/python.hpp>
#include <arrayobject.h>
#include <numpyconfig.h>
#include <numpy_interface.hpp>


using namespace boost::python;


#define LAPACK_SGGSVD sggsvd_
#define LAPACK_DGGSVD dggsvd_


extern "C" {
#define DEF_LAPACK_GGSVD(name, scalar_type) \
    void name(const char * jobu, const char * jobv, \
                       const char * jobq, const int * m, const int * n, \
                       const int * p, int * k, int * l, \
                       scalar_type * a, const int * lda, \
                       scalar_type * b, const int * ldb, \
                       scalar_type * alpha, \
                       scalar_type * beta, \
                       scalar_type * u, const int * ldu, \
                       scalar_type * v, const int * ldv, \
                       scalar_type * q, const int * ldq, \
                       scalar_type * work, int * iwork, \
                       int * info)

    DEF_LAPACK_GGSVD(LAPACK_SGGSVD, float);
    DEF_LAPACK_GGSVD(LAPACK_DGGSVD, double);
}

#define DEF_C_LAPACK_GGSVD(name, name_fortran, scalar_type) \
inline \
void name(const char jobu, const char jobv, const char jobq, \
    const int m, const int n, const int p, \
    int& k, int& l, \
    scalar_type* a, const int lda, \
    scalar_type* b, const int ldb, \
    scalar_type* alpha, \
    scalar_type* beta, \
    scalar_type* u, const int ldu, \
    scalar_type* v, const int ldv, \
    scalar_type* q, const int ldq, \
    scalar_type* work, int* iwork, \
    int& info) \
{ \
    name_fortran(&jobu, &jobv, &jobq, &m, &n, &p, \
                &k, &l, \
                a, &lda, \
                b, &ldb, \
                alpha, \
                beta, \
                u, &ldu, \
                v, &ldv, \
                q, &ldq, \
                work, iwork, \
                &info); \
} \

DEF_C_LAPACK_GGSVD(lapack_ggsvd, LAPACK_SGGSVD, float);
DEF_C_LAPACK_GGSVD(lapack_ggsvd, LAPACK_DGGSVD, double);


#define DEF_PYTHON_GGSVD(name, scalar_type) \
inline \
int name(const char jobu, const char jobv, const char jobq, \
           numeric::array& kl, \
           numeric::array& A, numeric::array& B, \
           numeric::array& alpha, numeric::array& beta, \
           numeric::array& U, numeric::array& V, numeric::array& Q, \
           numeric::array& work, numeric::array& iwork) \
{ \
    numpy_array<scalar_type> nA(A); \
    numpy_array<scalar_type> nB(B); \
    numpy_array<scalar_type> nalpha(alpha); \
    numpy_array<scalar_type> nbeta(beta); \
    numpy_array<scalar_type> nU(U); \
    numpy_array<scalar_type> nV(V); \
    numpy_array<scalar_type> nQ(Q); \
    numpy_array<scalar_type> nwork(work); \
    numpy_array<int> niwork(iwork); \
    numpy_array<int> nkl(kl); \
    int ierr; \
    lapack_ggsvd(jobu, jobv, jobq, nA.shape[0], nA.shape[1], \
          nB.shape[0], nkl[0], nkl[1], \
          nA.data, nA.leading_dimension, \
          nB.data, nB.leading_dimension, \
          nalpha.data, nbeta.data, \
          nU.data, nU.leading_dimension, \
          nV.data, nV.leading_dimension, \
          nQ.data, nQ.leading_dimension, \
          nwork.data, niwork.data, ierr); \
    return ierr; \
}

DEF_PYTHON_GGSVD(sggsvd, float);
DEF_PYTHON_GGSVD(dggsvd, double);


BOOST_PYTHON_MODULE(_ggsvd)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("sggsvd", sggsvd);
    def("dggsvd", dggsvd);
}