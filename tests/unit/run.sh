set -e

make -j
mkdir -p outputs
./unit_tests printout