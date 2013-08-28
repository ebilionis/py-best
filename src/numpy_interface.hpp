// Define a few routines to facilitate the use of numpy arrays from C.
//
// Author:
//      Ilias Bilionis
//
// Date:
//      11/28/2012

#include <boost/python.hpp>
#include <numpyconfig.h>


using namespace boost::python;



inline
char swap_trans(const char trans)
{
    return trans == 'N' ? 'T' : 'N';
}


template<typename scalar_type, int data_type, int inout_type>
struct numpy_array {

    numpy_array(numeric::array& a) :
        obj(PyArray_FROM_OTF(a.ptr(), data_type, inout_type)),
        data((scalar_type*)PyArray_DATA(obj)),
        shape(PyArray_DIMS(obj)),
        leading_dimension(static_cast<int>((
                        const scalar_type*)PyArray_GETPTR2(obj, 0, 1)
                        -(const scalar_type*)PyArray_GETPTR2(obj, 0, 0)))
    {}

    scalar_type& operator[](const int i) { return data[i]; }

    const scalar_type& operator[](const int i) const { return data[i]; }

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

