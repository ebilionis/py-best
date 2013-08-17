// Define a few routines to facilitate the use of numpy arrays from C.
//
// Author:
//      Ilias Bilionis
//
// Date:
//      11/28/2012

#include <boost/python.hpp>
#include <numpyconfig.h>


char swap_trans(const char trans)
{
    return trans == 'N' ? 'T' : 'N';
}

inline int leading_dimension(const PyObject* obj) {
    return static_cast<int>((const double*)PyArray_GETPTR2(obj, 0, 1)
                            - (const double*)PyArray_GETPTR2(obj, 0, 0));
}
