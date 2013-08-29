// Latin hypercube sampling
// Author: Ilias Bilionis
// Date: 12/2/2012

#include <latin_center_dataset.hpp>
#include <boost/python.hpp>
#include <numpyconfig.h>
#include <arrayobject.h>
#include <numpy_interface.hpp>

using namespace boost::python;

inline void lhs(numeric::array& X, int seed)
{
    numpy_array<double> nX(X);
    latin_center(nX.shape[0], nX.shape[1], &seed, nX.data);
}

BOOST_PYTHON_MODULE(_lhs)
{
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");

    def("lhs", lhs);
    def("get_seed", get_seed);
}
