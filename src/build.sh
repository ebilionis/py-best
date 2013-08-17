#!/bin/bash

FILES_TO_COMPILE="_ggsvd"

for file in $FILES_TO_COMPILE
do
    echo "Compiling file: ${file}.cpp"
    g++ -fPIC -shared -O3 ${file}.cpp -I./ -DHAVE_INLINE \
    -I$BOOST_DIR/include -L$BOOST_DIR/lib -I${PYTHON_INC_DIR} \
    -I${NUMPY_INC_DIR} -lboost_python -llapack\
    -o ../best/core/${file}.so
done
