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
    const PyObject* pA = PyArray_FROM_OTF(A.ptr(), NPY_FLOAT64,
                                          NPY_INOUT_FARRAY);
    double* dA = (double*)PyArray_DATA(pA);
    const npy_intp* dimsA = PyArray_DIMS(pA);
    const int ldA = leading_dimension(pA);
    const int m = dimsA[0];
    const int n = dimsA[1];

    // Data from B
    const PyObject* pB = PyArray_FROM_OTF(B.ptr(), NPY_FLOAT64,
                                          NPY_INOUT_FARRAY);
    double* dB = (double*)PyArray_DATA(pB);
    const npy_intp* dimsB = PyArray_DIMS(pB);
    const int ldB = leading_dimension(pB);
    const int p = dimsB[0];

    // Access to alpha
    const PyObject* palpha = PyArray_FROM_OTF(alpha.ptr(), NPY_FLOAT64,
                                              NPY_INOUT_FARRAY);
    double* dalpha = (double *)PyArray_DATA(palpha);

    // Access to beta
    const PyObject* pbeta = PyArray_FROM_OTF(beta.ptr(), NPY_FLOAT64,
                                              NPY_INOUT_FARRAY);
    double* dbeta = (double *)PyArray_DATA(pbeta);

    // Access to U
    const PyObject* pU = PyArray_FROM_OTF(U.ptr(), NPY_FLOAT64,
                                          NPY_INOUT_FARRAY);
    double* dU = (double*)PyArray_DATA(pU);
    const int ldU = leading_dimension(pU);

    // Access to V
    const PyObject* pV = PyArray_FROM_OTF(V.ptr(), NPY_FLOAT64,
                                          NPY_INOUT_FARRAY);
    double* dV = (double*)PyArray_DATA(pV);
    const int ldV = leading_dimension(pV);

    // Access to Q
    const PyObject* pQ = PyArray_FROM_OTF(Q.ptr(), NPY_FLOAT64,
                                          NPY_INOUT_FARRAY);
    double* dQ = (double*)PyArray_DATA(pQ);
    const int ldQ = leading_dimension(pQ);

    // Access to work
    const PyObject* pwork = PyArray_FROM_OTF(work.ptr(), NPY_FLOAT64,
                                              NPY_INOUT_FARRAY);
    double* dwork = (double *)PyArray_DATA(pwork);

    // Access to iwork
    const PyObject* piwork = PyArray_FROM_OTF(iwork.ptr(), NPY_INT32,
                                              NPY_INOUT_FARRAY);
    int* diwork = (int *)PyArray_DATA(piwork);

    // Access to kl
    const PyObject* pkl = PyArray_FROM_OTF(kl.ptr(), NPY_INT32,
                                              NPY_INOUT_FARRAY);
    int* dkl = (int *)PyArray_DATA(pkl);

    // Call the function
    int ierr;
    lapack_ggsvd(jobu, jobv, jobq, m, n, p, dkl[0], dkl[1],
          dA, ldA, dB, ldB,
          dalpha, dbeta, dU, ldU, dV, ldV, dQ, ldQ,
          dwork, diwork, ierr);

    return ierr;
}

BOOST_PYTHON_MODULE(_ggsvd)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("ggsvd", ggsvd);
}