// Wrapper for lapacks Incomplete Cholesky decomposition
// Author: Ilias Bilionis
// Date: 1/30/2012

#include <boost/python.hpp>
#include <numpyconfig.h>
#include <arrayobject.h>
#include <iostream>


using namespace boost::python;


extern "C" {
    void dpstrf_(const char* uplo, const int* n, double* ap,
                 const int* lda,
                 int* piv, int* rank, const double* tol,
                 double* work, int* info);
}


inline void pstrf(const char uplo, const int n, double* ap,
                  const int lda,
                  int* piv, int& rank, const double tol, double* work,
                  int& info)
{
    dpstrf_(&uplo, &n, ap, &lda, piv, &rank, &tol, work, &info);
}


inline void py_pstrf(const char uplo, numeric::array& a,
        numeric::array& piv, const double tol)
{
    PyObject* pa = PyArray_FROM_OTF(a.ptr(), NPY_FLOAT64,
            NPY_INOUT_FARRAY);
    double* da = (double* )PyArray_DATA(pa);
    const npy_intp* dimsa = PyArray_DIMS(pa);
    const int n = dimsa[0];
    PyObject* ppiv = PyArray_FROM_OTF(piv.ptr(), NPY_INT32,
            NPY_INOUT_FARRAY);
    const int lda = leadingdim(pa);
    int* dpiv = (int *)PyArray_DATA(ppiv);
    int ipiv[n];
    int rank;
    double work[2 * n];
    int info;
    pstrf(uplo, n, da, lda, dpiv, rank, tol, work, info);
    for(int i=0; i<n; i++)
        dpiv[i]--;
}


BOOST_PYTHON_MODULE(_dpstrf)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("pstrf", py_pstrf);
}
