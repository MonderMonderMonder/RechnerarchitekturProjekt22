#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include "header.h"


float log_2(float p) { // TODO: switch to integer and rename function log2_int
    if (p <= 0){
        fprintf(stderr, "Fehler! Eingabewerte dürfen nicht kleiner als 0 sein!\n");
        exit(EXIT_FAILURE);
    }
    if (p == 1)
        return 0; 
    int p_min = 1; 
    float acc = 0; 
    while (p_min<<1 <= p) {
        p_min <<= 1; 
        acc++; 
    }
    return acc; 
}

float log2_gcc_builtin(float p) { // TODO: switch to integer and rename function log2_gcc_builtin_int
    if (p <= 0)
        return -1;
    if (p == 1)
        return 0;
    return (float) (8*sizeof(unsigned int) - __builtin_clz((unsigned int)p) -1); // not processor specific, GCC finds instructions that will perform well for the architecture.
}

/**
 * @brief The integer part of log2(x). Returns le position of the leading 1, the lowest position being 0.
 * __builtin_clz(unsigned int) is not processor specific, GCC finds at compile time instructions that will perform well for the architecture.
 * @param x 
 * @return integer i for x = 2**i * (1 + f), 0 <= f < 1
 */
int leading_one_builtin(uint32_t x){
    if (x <= 0)
        return -1;
    return (int)(8*sizeof(uint32_t)) - __builtin_clz(x) -1; // (int)(8*sizeof(uint32_t)) = 32 at compile time
}


int leading_one_binarysearch(uint32_t x){
    if (x <= 0)
        return -1;
    int n = 0;
    uint32_t y;
    for (int i=16; i>=1; i>>=1) {
        y = x >> i; if (y != 0) { n += i; x = y;}
    }
    return n;
}

//naive implementation
float entropy_V1(size_t len, const float data[len]) {
    float copy[len];
    memcpy(copy, data, len*sizeof(float));
    qsort(copy, len, sizeof(float), cmp_floats_ascending);
    float sum = 0;
    for (size_t i=0; i<len; i++){
        sum+=copy[i];
    }
    if (sum < 0.9 || sum > 1.1){
        fprintf(stderr, "Summe der Wahrscheinlichkeiten ist ungleich 1! Eingabedatei auf Korrektheit überprüfen!\n");
        exit(EXIT_FAILURE);   
    }
    float result = 0; 
    float p;
    for (size_t i = 0; i < len; i++) { 
        p = data[i];
        if (log_2(1/p) == -1)
            return -1;
        result += p * log_2(1/p);
    }
    return result;
}

/*int main() {
    float a[5] = {0.2, 0.2, 0.2, 0.2, 0.2}; 
    printf("%.6f", entropy(5, a)); 
    
}*/
