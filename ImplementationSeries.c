#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include "header.h"

float pow_naive(float x, int e) {
    if (e == 0)
        return 1;
    if (e == 1)
        return x;
    float result = 1; 
    for (int j = e; j > 0; j--)
        result = result * x;
    return result;
}

//instead of multiplying the value of x with itself in e iterations, if the exponent is bigger than 4 have a value be x^4 and multiply it constantly with itself until the exponent is reached, and then proceed as as in the naive implementation. This lowers the amount of iterations for bigger exponents.
float pow_optimized(float x, int e) {
    if (e == 0)
        return 1;
    if (e == 1)
        return x; 
    int j = e; 
    float result = 1;
    if (j >= 4) {
        float a = x*x*x*x;
        for (j; j >= 4; j-=4) {
            result *= a; 
        }
    }
    if (j < 4) {
        for (j; j > 0; j--) {
        result = result * x;
        }
        return result; 
    }
    return result; 
}

float ln(float x, int n) {
    float result = 0;
    // TODO: add the results of 2 iterations at the same time
    for (int i = 1; i <= n; i++)
        result = (i % 2 == 0)? result + pow_naive(x-1, i)/i : result - pow_naive(x-1, i)/i;
    return result;
}

//Calculates ln of a number with the use of intrinsics. 
float ln_simd_simple(float x, int n) {
    if (x == 0) {
        fprintf(stderr, "Fehler! Eingaben müssen ungleich 0 sein!\n");
        exit(EXIT_FAILURE);
    }
    int i = 0; 
    float result = 0; 
    if (n >= 4) {
        __m128 result_simd = _mm_setzero_ps();  
        float pow_optimized_array[4];
        for (i; i+4 <= n; i+=4) {
            pow_optimized_array[0] = pow_optimized(x - 1, i+1)/(i+1); 
            pow_optimized_array[1] = pow_optimized(x-1, i+2)/(i+2); 
            pow_optimized_array[2] = pow_optimized(x-1, i+3)/(i+3);
            pow_optimized_array[3] = pow_optimized(x-1, i+4)/(i+4); 
            __m128 pow_optimized_simd  = _mm_loadu_ps (pow_optimized_array); 
            result_simd = _mm_add_ps(result_simd, pow_optimized_simd); 
        }
        result = result_simd[0] - result_simd[1] + result_simd[2] - result_simd[3]; 
    }
    if (n < 4) {
        i = n; 
    }
        if (i % 4 >= 1) {
            result += pow_optimized(x - 1, i+1)/i+1; 
        }
        if (i % 4 >= 2) {
            result += -pow_optimized(x-1, i+2)/i+2;
        }
        if (i % 4 == 3) {
            result += pow_optimized(x-1, i+3)/i+3; 
        }
        return result; 
    }
    
//same as ln_simd_simple but with an xmm register as a parameter and return type. 
__m128 ln_simd_complex(__m128 x, int n) {
    float result[4]; 
    for (int i = 0; i < 4; i++) {
        result[i] = ln_simd_simple(x[i], n); 
    }
    __m128 result_simd = _mm_loadu_ps (result);
    return result_simd; 
}

//basic implementation of log of base two. 
float log2_serie(float p, int n) {
    if (p == 0)
        return -1;
    return ln(p, n)/ln(2, n); 
}

//calculates log of base two using the ln_simd_simple function. 
float log2_serie_simd_simple(float p, int n) {
    if (p == 0) {
        fprintf(stderr, "Fehler! Eingaben müssen ungleich 0 sein!\n");
        exit(EXIT_FAILURE);
    }
    return ln_simd_simple(p, n)/ln_simd_simple(2, n); 
}

//same as log2_simd_simple, but using an xmm register as a parameter and return type for the improvement of the entropy function. (See entropy_simd)
__m128 log2_serie_simd_complex(__m128 p, int n) {
    __m128 result = ln_simd_complex(p, n); 
    for (int i = 0; i<4; i++) {
        result[i] = result[i] / ln_simd_simple(2, n); 
    } 
    return result; 
}

float entropy_V2(size_t len, const float data[len]) {
    float p;
    float psum = 0;
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        p = data[i];
        if (log2_serie(p, 1000) == -1)
            return -1; 
        result -= p * log2_serie(p, 1000);
        psum += p;
    }
    if (psum < 0.9 || psum > 1.1){ // if we calculate sum separately, we access array 2 times more (performance loss)
        fprintf(stderr, "Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheitüberprüfen!\n");
        exit(EXIT_FAILURE);   
    }
    return result; 
}

//calculates entropy using simd to improve perfomance. Psum and the addition of all the probabilities is improved through this use of intrinsics. 
float entropy_V4(size_t len, const float data[len]) {
    size_t i = 0; 
    float p; 
    float psum = 0; 
    size_t length = len; 
    __m128 psum_simd = _mm_setzero_ps(); 
    __m128 log2_array; 
    __m128 result2 = _mm_setzero_ps(); 
    float result = 0; 
    if (length >= 4) {
    for (i; length >= 4; length-=4) {
        if (log2_serie_simd_simple(data[i], 1000) == -1) {
            return -1; 
        }
        __m128 b = _mm_loadu_ps (data + i);
        psum_simd = _mm_add_ps(psum_simd, b); 
        log2_array = log2_serie_simd_complex(b, 1000);  
        b = _mm_mul_ps(b, log2_array); 
        result2 = _mm_sub_ps(result2, b); 
        i+=4; 
    }
    result = (-result2[0] - result2[1] - result2[2] - result2[3]) * (-1); 
    psum += psum_simd[0] + psum_simd[1] + psum_simd[2] + psum_simd[3]; 
    } 

    if (length < 4) {
        for (int j = i; j < len; j++) {
        p = data[j];
        if (log2_serie_simd_simple(p, 1000) == -1)
            return -1; 
        result -= p * log2_serie(p, 1000);
        psum += p;
    }
    }
    if (psum < 0.9 || psum > 1.1){
        fprintf(stderr, "Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheit überprüfen!\n");
        exit(EXIT_FAILURE);   
    }
    return result; 
    }





/**
 * @brief Helper function printing the mean absolute deviation of our serie implementation in comparision to the gcc log2 function, 
 * and the execution time
 * 
 */
void precision_and_perf () {
    float sum, avg;
    float t;
    struct timespec start, end;
    for (int iter = 150; iter<=250; iter++) {
        // absolute difference average
        sum = 0;
        for (float x=0.001; x<=1; x+=0.001, sum+=abs(log2_serie_simd_simple(x, iter)-log2f(x))) {}
        avg = sum / 1000;
        // execution time
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (float x=0.001; x<=1; x+=0.001, log2_serie(x, iter)) {}
        clock_gettime(CLOCK_MONOTONIC, &end);
        t = end.tv_sec-start.tv_sec+(1e-9)*(end.tv_nsec-start.tv_nsec);
        printf("log2_serie %d iterations:\t%.8a    %.16f\n", iter, avg, t);
    }
    return;
}



/*int main (int argc , char ** argv ){ 
    precision_and_perf(); 
    }*/
