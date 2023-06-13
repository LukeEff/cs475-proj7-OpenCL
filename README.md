# Project 7: Fourier Analysis using MPI

## OpenCL Implementation

The OpenCL implementation of the project is an adaptation of the OpenMPI implementation. At a high level, the
OpenCL implementation is similar to the OpenMPI implementation, except that in the OpenCL problem space, far more
processors are available. Since more processors are available, I decided to throw out the "Per-Processor" concept and
assign each GPU core a single element in the signal array. It computes MAXPERIODS number of sums, and then writes the
result to the output array. The output array is then copied back to the host, and the host will write the output to
the output file.

In order to avoid race conditions, I atomically add the result of each sum to the output array. This is an expensive
operation, especially since every single GPU core would have to access the same memory location for each sum. By
performing per-work-group reductions, the number of atomic operations per work group is reduced from MAXPERIODS * LOCALSIZE
to MAXPERIODS and skyrockets the performance of the program. 

### Possible Optimizations

* Re-introduce PPSize and find a goldilocks zone for amount of signal elements per processor.