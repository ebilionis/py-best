PYTHON_INC_DIR=/usr/include/python2.7
NUMPY_INC_DIR=/usr/include/python2.7/numpy
FILES_TO_COMPILE="_lhs "

for file in $FILES_TO_COMPILE
do
    echo "Compiling: ${file}.cpp"
    g++ -fPIC -shared -O3 ${file}.cpp -I./ -DHAVE_INLINE \
        -I$BOOST_DIR/include -L$BOOST_DIR/lib -I${PYTHON_INC_DIR}\
        -I${NUMPY_INC_DIR} -lboost_python $MKL -lmkl_mc -o ../best/core/${file}.so
done

