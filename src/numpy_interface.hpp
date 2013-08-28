// Define a few routines to facilitate the use of numpy arrays from C.
//
// Author:
//      Ilias Bilionis
//
// Date:
//      11/28/2012

#include <boost/python.hpp>
#include <boost/type_traits.hpp>
#include <numpyconfig.h>


using namespace boost::python;
using boost::remove_const;
using boost::true_type;
using boost::is_const;


inline
char swap_trans(const char trans)
{
    return trans == 'N' ? 'T' : 'N';
}

/// Turn a type to a numpy type.
template<typename scalar_type>
struct to_numpy_type
{
    static const int value = NPY_OBJECT;
};

#define DEF_TO_NUMPY_TYPE(scalar_type, numpy_type) \
template<> \
struct to_numpy_type<scalar_type> \
{ \
    static const int value = numpy_type; \
}

DEF_TO_NUMPY_TYPE(bool, NPY_BOOL);
DEF_TO_NUMPY_TYPE(float, NPY_FLOAT32);
DEF_TO_NUMPY_TYPE(double, NPY_FLOAT64);
DEF_TO_NUMPY_TYPE(int, NPY_INT32);

template<typename bool_type>
struct to_numpy_in_out_base {
    static const int value = NPY_INOUT_FARRAY;
};

template<>
struct to_numpy_in_out_base<true_type> {
    static const int value = NPY_IN_FARRAY;
};

template<typename scalar_type>
struct to_numpy_in_out : public to_numpy_in_out_base<is_const<scalar_type> >
{};


/// This is a structure that serves as a C++ interface for numpy arrays.
template<typename scalar_type>
struct numpy_array {

    numpy_array(numeric::array& a) :
        obj(PyArray_FROM_OTF(a.ptr(),
        to_numpy_type<typename remove_const<scalar_type>::type >::value,
        to_numpy_in_out<scalar_type>::value)),
        data((scalar_type*)PyArray_DATA(obj)),
        shape(PyArray_DIMS(obj)),
        leading_dimension(static_cast<int>((
        const typename remove_const<scalar_type>::type*)PyArray_GETPTR2(obj, 0, 1)
        -(const typename remove_const<scalar_type>::type*)PyArray_GETPTR2(obj, 0, 0)))
    {}

    scalar_type& operator[](const int i) { return data[i]; }

    const typename remove_const<scalar_type>::type& operator[](const int i) const
    { return data[i]; }

    const PyObject* obj;

    scalar_type* data;

    const npy_intp* shape;

    const int leading_dimension;
};


inline
int leading_dimension(const PyObject* obj) {
    return static_cast<int>((const double*)PyArray_GETPTR2(obj, 0, 1)
                            - (const double*)PyArray_GETPTR2(obj, 0, 0));
}

