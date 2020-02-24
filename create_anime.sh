export CXX=g++-9
rm -r build
mkdir build
cd build
cmake -DCODE_GENERATION=ON -DBUILD_DOCUMENTATION=ON -DCMAKE_INSTALL_PREFIX=../../install/ ..
