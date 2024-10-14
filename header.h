#ifndef FOO_DOT_H
#define FOO_DOT_H

float log2_serie(float p, int n);
float ln(float p, int loops);
float potenz(float n, int m);

float entropy(size_t len, const float data[len]);
float entropy_V1(size_t len, const float data[len]);
float entropy_V2(size_t len, const float data[len]);
float entropy_V3(size_t len, const float data[len]);
float entropy_V4(size_t len, const float data[len]);
float entropy_V5(size_t len, const float data[len]);
float entropy_V6(size_t len, const float data[len]);
float entropy_V7(size_t len, const float data[len]);
int cmp_floats_ascending(const void *a, const void *b);
void precision_and_perf();

#endif