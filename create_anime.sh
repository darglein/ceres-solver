export CXX=g++-9
rm -r build
mkdir build
cd build
cmake -DCODE_GENERATION=ON ..
