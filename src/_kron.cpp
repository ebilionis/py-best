#include <boost/python.hpp>
#include <string>

#include <numpyconfig.h>
#include <arrayobject.h>

using namespace boost::python;

#define BLAS_DTRSM dtrsm_
#define BLAS_DTRMM dtrmm_
#define BLAS_DGEMM dgemm_

// MKL/BLAS interface
extern "C" {
    void BLAS_DTRSM(const char* side, const char* uplo, const char* transa,
                    const char* diag, const int* m, const int* n,
                    double const* alpha, double const* a, int const* lda,
                    double* b, int const* ldb);
}

extern "C" {
    void BLAS_DTRMM(const char* side, const char* uplo, const char* transa,
                    const char* diag, const int* m, const int* n,
                    double const* alpha, double const* a, int const* lda,
                    double* b, int const* ldb);
}

extern "C" {
    void BLAS_DGEMM(const char* transa, const char* transb,
                    const int* m, const int* n, const int* k,
                    const double* alpha, const double* a, const int* lda,
                    const double* b, const int* ldb, const double* beta,
                    double* c, const int* ldc);
}

// Solve the linear system A * X = Y, when A is a triangular matrix,
// in place.
inline void trsm(const char side, const char uplo, const char transa,
                 const char diag, const int m, const int n,
                 const double alpha, const double* a, const int lda,
                 double* b, const int ldb)
{
    BLAS_DTRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda,
               b, &ldb);
}

// Multiply by a triangular matrix in place.
inline void trmm(const char side, const char uplo, const char transa,
                 const char diag, const int m, const int n,
                 const double alpha, const double* a, const int lda,
                 double* b, const int ldb)
{
    BLAS_DTRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda,
               b, &ldb);
}

inline void gemm(const char transa, const char transb,
                 const int m, const int n, const int k,
                 const double alpha, const double* a, const int lda,
                 const double* b, const int ldb, const double beta,
                 double* c, const int ldc)
{
    BLAS_DGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
               &beta, c, &ldc);
}

char swap_trans(const char trans)
{
    return trans == 'N' ? 'T' : 'N';
}

inline int leading_dimension(const PyObject* obj) {
    return static_cast<int>((const double*)PyArray_GETPTR2(obj, 0, 1)
                            - (const double*)PyArray_GETPTR2(obj, 0, 0));    
}

// Solve the linear system A1 x A2 x ... x Ak * X = Y
// k is the number of matrices in the Kronecker product.
// uplo is a list of dimension k specifying the storage type of each matrix
// ('U' for upper and 'L' for lower).
// trans is a list of dimension k specifying whether or not each matrix in
// needs to be transposed ('T' for transposed 'N' otherwise).
inline void kron_trsm(const int k, const char* uplo, const char* trans,
                      const int* m, const int n,
                      const double** a, const int* lda,
                      double* b, const int ldb)
{
    // Loop over columns
    for(int j=0; j<n; j++) {
        double* b_j = b + j * ldb;
        // Solve the linear system B_j * A[0]^T
        trsm('R', uplo[0], swap_trans(trans[0]), 'N',
             ldb / m[0], m[0], 1., a[0], lda[0],
             b_j, ldb / m[0]);
    }
    if(k == 1)
        return;
    for(int j=0; j<n; j++) {
        double* b_j = b + j * ldb;
        kron_trsm(k-1, uplo+1, trans+1, m+1, m[0], a+1, lda+1,
                  b_j, ldb / m[0]);
    }
}

inline void kron_trmm(const int k, const char* uplo, const char* trans,
                      const int* m, const int n,
                      const double** a, const int* lda,
                      double* b, const int ldb)
{
    // Loop over columns
    for(int j=0; j<n; j++) {
        double* b_j = b + j * ldb;
        // Solve the linear system B_j * A[0]^T
        trmm('R', uplo[0], swap_trans(trans[0]), 'N',
             ldb / m[0], m[0], 1., a[0], lda[0],
             b_j, ldb / m[0]);
    }
    if(k == 1)
        return;
    for(int j=0; j<n; j++) {
        double* b_j = b + j * ldb;
        kron_trmm(k-1, uplo+1, trans+1, m+1, m[0], a+1, lda+1,
                  b_j, ldb / m[0]);
    }
    std::cout << std::endl;
}

// Python interface
inline void py_trsm(const char side, const char uplo, const char transa,
                    const char diag, const double alpha,
                    const numeric::array& a, numeric::array& b)
{
    const PyObject* pa = PyArray_FROM_OTF(a.ptr(), NPY_FLOAT64,
                                          NPY_IN_FARRAY);
    const double* da = (const double*)PyArray_DATA(pa);
    PyObject* pb = PyArray_FROM_OTF(b.ptr(), NPY_FLOAT64,
                                    NPY_INOUT_FARRAY);
    double* db = (double*)PyArray_DATA(pb);
    const npy_intp* dimsb = PyArray_DIMS(pb);
    const int m = dimsb[0];
    const int n = dimsb[1];
    const int lda = leading_dimension(pa);
    const int ldb = leading_dimension(pb);
    trsm(side, uplo, transa, diag, m, n, alpha, da, lda, db, ldb);
}

inline void py_trmm(const char side, const char uplo, const char transa,
                    const char diag, const double alpha,
                    const numeric::array& a, numeric::array& b)
{
    const PyObject* pa = PyArray_FROM_OTF(a.ptr(), NPY_FLOAT64,
                                          NPY_IN_FARRAY);
    const double* da = (const double*)PyArray_DATA(pa);
    const npy_intp* dimsa = PyArray_DIMS(pa);
    PyObject* pb = PyArray_FROM_OTF(b.ptr(), NPY_FLOAT64,
                                    NPY_INOUT_FARRAY);
    double* db = (double*)PyArray_DATA(pb);
    const npy_intp* dimsb = PyArray_DIMS(pb);
    const int m = dimsb[0];
    const int n = dimsb[1];
    const int lda = leading_dimension(pa);
    const int ldb = leading_dimension(pb);
    trmm(side, uplo, transa, diag, m, n, alpha, da, lda, db, ldb);
}

inline void py_kron_trsm(const std::string& uplo, const std::string& trans,
                         const tuple& ta, numeric::array& b)
{
    const PyObject* a = ta.ptr();
    const int k = PyTuple_GET_SIZE(a);
    const double** da = new const double*[k];
    int* m = new int[k];
    int* lda = new int[k];
    for(int i=0; i<k; i++) {
        PyObject* aa = PyTuple_GET_ITEM(a, i);
        const PyObject* pa = PyArray_FROM_OTF(aa, NPY_FLOAT64,
                                              NPY_IN_FARRAY);
        da[i] = (const double*)PyArray_DATA(pa);
        const npy_intp* dimsa = PyArray_DIMS(pa);
        m[i] = dimsa[0];
        lda[i] = leading_dimension(pa);
    }
    PyObject* pb = PyArray_FROM_OTF(b.ptr(), NPY_FLOAT64,
                                    NPY_INOUT_FARRAY);
    double* db = (double*)PyArray_DATA(pb);
    const npy_intp* dimsb = PyArray_DIMS(pb);
    const int n = dimsb[1];
    const int ldb = leading_dimension(pb);
    kron_trsm(k, uplo.c_str(), trans.c_str(), m, n, da, lda, db, ldb);
    delete da;
    delete m;
    delete lda;
}

inline void py_kron_trmm(const std::string& uplo, const std::string& trans,
                         const tuple& ta, numeric::array& b)
{
    const PyObject* a = ta.ptr();
    const int k = PyTuple_GET_SIZE(a);
    const double** da = new const double*[k];
    int* m = new int[k];
    int* lda = new int[k];
    for(int i=0; i<k; i++) {
        PyObject* aa = PyTuple_GET_ITEM(a, i);
        const PyObject* pa = PyArray_FROM_OTF(aa, NPY_FLOAT64,
                                              NPY_IN_FARRAY);
        da[i] = (const double*)PyArray_DATA(pa);
        const npy_intp* dimsa = PyArray_DIMS(pa);
        m[i] = dimsa[0];
        lda[i] = leading_dimension(pa);
    }
    PyObject* pb = PyArray_FROM_OTF(b.ptr(), NPY_FLOAT64,
                                    NPY_INOUT_FARRAY);
    double* db = (double*)PyArray_DATA(pb);
    const npy_intp* dimsb = PyArray_DIMS(pb);
    const int n = dimsb[1];
    const int ldb = leading_dimension(pb);
    kron_trmm(k, uplo.c_str(), trans.c_str(), m, n, da, lda, db, ldb);
    delete da;
    delete m;
    delete lda;
}

inline void py_gemm(const char transa, const char transb,
                    const double alpha, const numeric::array& a,
                    const numeric::array& b, const double beta,
                    numeric::array& c)
{
    const PyObject* pa = PyArray_FROM_OTF(a.ptr(), NPY_FLOAT64,
                                          NPY_IN_FARRAY);
    const double* da = (const double*)PyArray_DATA(pa);
    const npy_intp* dimsa = PyArray_DIMS(pa);
    const PyObject* pb = PyArray_FROM_OTF(b.ptr(), NPY_FLOAT64,
                                          NPY_IN_FARRAY);
    const double* db = (const double*)PyArray_DATA(pb);
    PyObject* pc = PyArray_FROM_OTF(c.ptr(), NPY_FLOAT64,
                                    NPY_INOUT_FARRAY);
    double* dc = (double*)PyArray_DATA(pc);
    const npy_intp* dimsc = PyArray_DIMS(pc);
    const int lda = leading_dimension(pa);
    const int ldb = leading_dimension(pb);
    const int ldc = leading_dimension(pc);
    const int m = dimsc[0];
    const int n = dimsc[1];
    const int k = transa == 'N' ? dimsa[1] : dimsa[0];
    gemm(transa, transb, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc);
}

BOOST_PYTHON_MODULE(_kron)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("trsm", py_trsm);
    def("trmm", py_trmm);
    def("kron_trsm", py_kron_trsm);
    def("kron_trmm", py_kron_trmm);
    def("gemm", py_gemm);
}

