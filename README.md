```
git clone git@github.com:acreyes/advection_demo.git --recursive
mkdir -p advection_demo/build
cd advection_demo/build
cmake ..
cmake --build . --parallel
```

Parthenon wants parallel HDF5 & MPI and kokkos will look for openmp. It might help to add the environment variables
```
export HDF5_ROOT=/path/to/hdf5
export CXX=/path/to/mpicxx
export CC=/path/to/mpicc
```
