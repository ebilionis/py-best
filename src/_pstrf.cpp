// Wrapper for lapacks Incomplete Cholesky decomposition
// Author: Ilias Bilionis
// Date: 1/30/2012

#include <boost/python.hpp>
#include <numpyconfig.h>
#include <arrayobject.h>
#include <numpy_interface.hpp>


using namespace boost::python;


#define LAPACK_SPSTRF spstrf_
#define LAPACK_DPSTRF dpstrf_


extern "C" {

#define DEF_LAPACK_PSTRF(name, scalar_type) \
    void name(const char* uplo, const int* n, scalar_type* ap, \
              const int* lda, \
              int* piv, int* rank, const scalar_type* tol, \
              scalar_type* work, int* info)

    DEF_LAPACK_PSTRF(LAPACK_SPSTRF, float);
    DEF_LAPACK_PSTRF(LAPACK_DPSTRF, double);
}


#define DEF_C_LAPACK_PSTRF(name, name_fortran, scalar_type) \
inline void name(const char uplo, const int n, scalar_type* ap, \
                 const int lda, \
                 int* piv, int& rank, const scalar_type tol, \
                 scalar_type* work, \
                 int& info) \
{ \
    name_fortran(&uplo, &n, ap, &lda, piv, &rank, &tol, work, &info); \
}

DEF_C_LAPACK_PSTRF(lapack_pstrf, LAPACK_SPSTRF, float);
DEF_C_LAPACK_PSTRF(lapack_pstrf, LAPACK_DPSTRF, double);


#define DEF_PYTHON_PSTRF(name, scalar_type) \
inline \
int name(const char uplo, numeric::array& a, numeric::array& piv, \
         numeric::array& rank, \
         const scalar_type tol, \
         numeric::array& work) \
{ \
    numpy_array<scalar_type> na(a); \
    numpy_array<int> npiv(piv); \
    numpy_array<int> nrank(rank); \
    numpy_array<scalar_type> nwork(work); \
    int info; \
    lapack_pstrf(uplo, na.shape[0], na.data, na.leading_dimension, \
                 npiv.data, nrank[0], tol, nwork.data, info); \
    for(int i=0; i<na.shape[0]; i++) \
        npiv.data[i]--; \
    return info; \
}

DEF_PYTHON_PSTRF(spstrf, float);
DEF_PYTHON_PSTRF(dpstrf, double);


BOOST_PYTHON_MODULE(lib_pstrf)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("spstrf", spstrf);
    def("dpstrf", dpstrf);
}