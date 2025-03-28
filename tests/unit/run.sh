set -e

make clean
make -j
mkdir -p outputs
./unit_tests printout