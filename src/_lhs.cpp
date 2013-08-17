// Latin hypercube sampling
// Author: Ilias Bilionis
// Date: 12/2/2012

#include <latin_center_dataset.hpp>
#include <boost/python.hpp>
#include <numpyconfig.h>
#include <arrayobject.h>

using namespace boost::python;

inline void lhs(numeric::array& X, int seed)
{
    const PyObject* pX = PyArray_FROM_OTF(X.ptr(), NPY_FLOAT64,
                                          NPY_OUT_FARRAY);
    double* dX = (double*)PyArray_DATA(pX);
    const npy_intp* dimsX = PyArray_DIMS(pX);
    latin_center(dimsX[0], dimsX[1], &seed, dX); 
}

BOOST_PYTHON_MODULE(_lhs)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("lhs", lhs);
    def("get_seed", get_seed);
}
