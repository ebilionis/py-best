#include <boost/python.hpp>
#include <arrayobject.h>
#include <numpyconfig.h>
#include <numpy_interface.hpp>


using namespace boost::python;


#define LAPACK_DGGSVD dggsvd_


extern "C" {
    void LAPACK_DGGSVD(const char * jobu, const char * jobv,
                       const char * jobq, const int * m, const int * n,
                       const int * p, int * k, int * l,
                       double * a, const int * lda,
                       double * b, const int * ldb,
                       double * alpha,
                       double * beta,
                       double * u, const int * ldu,
                       double * v, const int * ldv,
                       double * q, const int * ldq,
                       double * work, int * iwork,
                       int * info);
}


inline
void lapack_ggsvd(const char jobu, const char jobv, const char jobq,
    const int m, const int n, const int p,
    int& k, int& l,
    double* a, const int lda,
    double* b, const int ldb,
    double* alpha,
    double* beta,
    double* u, const int ldu,
    double* v, const int ldv,
    double* q, const int ldq,
    double* work, int* iwork,
    int& info)
{
    LAPACK_DGGSVD(&jobu, &jobv, &jobq, &m, &n, &p,
                &k, &l,
                a, &lda,
                b, &ldb,
                alpha,
                beta,
                u, &ldu,
                v, &ldv,
                q, &ldq,
                work, iwork,
                &info);
}


inline
int ggsvd(const char jobu, const char jobv, const char jobq,
           numeric::array& kl,
           numeric::array& A, numeric::array& B,
           numeric::array& alpha, numeric::array& beta,
           numeric::array& U, numeric::array& V, numeric::array& Q,
           numeric::array& work, numeric::array& iwork)
{
    // Data from A
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nA(A);

    // Data from B
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nB(B);

    // Access to alpha
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nalpha(alpha);

    // Access to beta
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nbeta(beta);

    // Access to U
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nU(U);

    // Access to V
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nV(V);

    // Access to Q
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nQ(Q);

    // Access to work
    numpy_array<double, NPY_FLOAT64, NPY_INOUT_FARRAY> nwork(work);

    // Access to iwork
    numpy_array<int, NPY_INT32, NPY_INOUT_FARRAY> niwork(iwork);

    // Access to kl
    numpy_array<int, NPY_INT32, NPY_INOUT_FARRAY> nkl(kl);

    // Call the function
    int ierr;
    lapack_ggsvd(jobu, jobv, jobq, nA.shape[0], nA.shape[1],
          nB.shape[0], nkl[0], nkl[1],
          nA.data, nA.leading_dimension,
          nB.data, nB.leading_dimension,
          nalpha.data, nbeta.data,
          nU.data, nU.leading_dimension,
          nV.data, nV.leading_dimension,
          nQ.data, nQ.leading_dimension,
          nwork.data, niwork.data, ierr);

    return ierr;
}

BOOST_PYTHON_MODULE(lib_ggsvd)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("ggsvd", ggsvd);
}