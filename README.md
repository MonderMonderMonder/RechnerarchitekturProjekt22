# Fast Information Entropy Calculation

This repository implements various approaches to calculate Shannon's entropy. The implementations prioritize performance, accuracy, and efficient memory usage. Multiple methods are implemented in C, including naive and optimized techniques using lookup tables (LUT) and series expansions with SIMD and Intel SSE intrinsics.

## Contents

1. **`ImplementationNaive.c`** - Naive entropy calculation.
2. **`ImplementationSeries.c`** - Uses a series expansion with SIMD optimizations.
3. **`ImplementationLUT.c`** - Optimized entropy calculation using precomputed LUT and SIMD.
4. **`SupportingProgram.c`** - Supplementary functions for shared functionality.
5. **`header.h`** - Header file with function declarations and macros.
6. **`Makefile`** - Build instructions.
7. **`report.pdf`** - Detailed documentation covering theory, methods, performance, and accuracy evaluations.

## Background

Entropy, as defined by Claude Shannon, quantifies the uncertainty or information content within a source. This project calculates entropy by evaluating the probability distribution of symbols, where the information content of each symbol is inversely proportional to its occurrence probability.

## Approaches

### 1. Naive Implementation (`ImplementationNaive.c`)

Serves as a baseline, without any optimizations.

### 2. Series Expansion (`ImplementationSeries.c`)

Improves accuracy using a Taylor series expansion along with SIMD optimizations for multiple calculations in parallel thanks to `__m128` data types and corresponding Intel SSE Intrinsics, allowing four single-precision floating-point operations simultaneously.

Faster than the naive approach with a significant performance boost for large datasets due to SIMD.

### 3. Lookup Table (LUT) Implementation (`ImplementationLUT.c`)

Evenly and unevenly spaced LUTs stores precomputed logarithmic values for fast approximations, using linear or polynomial interpolation for better accuracy. It is also enhanced with SIMD.

SSE registers store adjacent values from the LUT, and SIMD instructions perform interpolation in a single operation.

Due to irregular memory access patterns in unevenly spaced LUTs, the SIMD implementation has certain limitations with binary search operations in the LUT.

This method delivers with SIMD the best performance for large datasets. However, unevenly spaced LUTs require more complex handling, reducing some SIMD gains.

### Why SIMD?

SIMD (Single Instruction, Multiple Data) is a technique where a single instruction processes multiple data points simultaneously. Intel's SSE (Streaming SIMD Extensions) intrinsics allow fine-grained control over SIMD operations at the instruction level, which optimizes loops and improves execution time by minimizing redundant calculations.

Functions are restructured to handle data in chunks, where `__m128` SSE data types allow four floating-point values to be processed in parallel.
SSE registers load multiple LUT entries simultaneously, improving memory bandwidth efficiency and eliminating separate computations for each entry.
When appropriate, loop unrolling with SSE intrinsics reduces branch overhead, making the code run faster, especially for long series calculations.

## Building and Running

1. **Compilation**: Use the provided `Makefile` to compile and link all implementations:
   ```bash
   make all
   ```

2. **Usage**:
   ```bash
   ./main -h
    Usage: [-h | --help] | [-V version] [-B repetitions]
     -h:     Displays this help message and an example of usage
     -help:  Displays this help message and an example of usage
     -V:     version (0 - 7) Specifies the implementation version to use
     -B:     repetitions (1 - 2147483647) Measures runtime, with the optional argument indicating the number of repetitions
    
       Versions:
           -V0 - LUT (uneven, polynomial)
           -V1 - naive
           -V2 - Series
           -V3 - LUT-SIMD (uneven, polynomial)
           -V4 - Series-SIMD
           -V5 - LUT (uneven, linear)
           -V6 - LUT (even, polynomial)
           -V7 - LUT (even, linear)
    
    Example usage:
                 [./main] -V1 -B3
     -V1   --> uses the first comparison implementation (2nd implementation)
     -B    --> runtime is measured and displayed
     -B3   --> function call is repeated 3 times

   ```

3. **Cleanup**:
   ```bash
   make clean
   ```

## Documentation

Refer to `report.pdf` for in-depth explanations of entropy theory, each implementation, performance benchmarking, and accuracy evaluations.
