#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mkl.h>
#include <omp.h>
#include "immintrin.h"

int NUM_THREADS;

template<typename Float>
void print_vector(Float *x, int len, char *name = "vector")
{
	printf("%s:", name);
	for (int i = 0; i < len; ++i) {
		//printf(" %.20Lf", x[i]);
		std::cout << " " << x[i];
	}
	putchar('\n');
}

template<typename Float>
Float iblas_dot(int n, Float *x, Float *y)
{
	Float dtemp = 0.0;
	Float ddot = 0.0;
	if (n <= 0) {
		return ddot;
	}

	int m = n % 5;
	if (m != 0) {
		for (int i = 0; i < m; ++i) {
			dtemp += x[i] * y[i];
		}
		if (n < 5) {
			ddot = dtemp;
			return ddot;
		}
	}
	for (int i = m; i < n; i += 5) {
		dtemp += x[i] * y[i]
				+ x[i + 1] * y[i + 1]
				+ x[i + 2] * y[i + 2]
				+ x[i + 3] * y[i + 3]
				+ x[i + 4] * y[i + 4];
	}
	ddot = dtemp;
	return ddot;
}

//template<typename Float>
//Float idot(int n, Float *x, Float *y)
//{
//	Float dot = 0.0;
//#pragma omp parallel for reduction(+: dot)
//	for (int i = 0; i < n; ++i) {
//		dot += x[i] * y[i];
//	}
//
//	return dot;
//}

double idot(int n, double *x, double *y)
{
	double sum = 0.0;
	int num_packed_double = 8; // number of double numbers in SIMD a register
	int remainder = n % num_packed_double;
	int bound = n - remainder;

	for (int i = 0; i < bound; i += num_packed_double) {
		__m512d x_v = _mm512_load_pd(x + i);
		__m512d y_v = _mm512_load_pd(y + i);
		__m512d t_v = _mm512_mul_pd(x_v, y_v);
		sum += _mm512_reduce_add_pd(t_v);
	}

	// Add the remainder
	unsigned char t_m = (unsigned char) 0xFF >> (num_packed_double - remainder);
	__mmask8 r_m = (__mmask8) t_m;
	__m512d x_v = _mm512_mask_load_pd(_mm512_undefined_pd(), r_m, x + bound);
	__m512d y_v = _mm512_mask_load_pd(_mm512_undefined_pd(), r_m, y + bound);
	__m512d t_v = _mm512_mask_mul_pd(_mm512_undefined_pd(), r_m, x_v, y_v);
	sum += _mm512_mask_reduce_add_pd(r_m, t_v);
	return sum;
}

void intput(int n, long double *x, long double *y)
{
	srand(time(0));
//#pragma omp parallel for num_threads(64)
	for (int i = 0; i < n; ++i) {
		x[i] = (long double) rand()/RAND_MAX;
		y[i] = (long double) rand()/RAND_MAX;
	}
}

long double kahan_dot(int n, long double *x, long double *y)
{
	long double sum = 0.0;
	long double c = 0.0;

	for (int i = 0; i < n; ++i) {
		long double r = x[i] * y[i] - c;
		long double t = sum + r;
		c = (t - sum) - r;
		sum = t;
	}

	return sum;
}

void test()
{
	double x[5] __attribute__((aligned(64))) = {1, 1, 1, 1, 1};
	double y[5] __attribute__((aligned(64))) = {1, 1, 1, 1, 1};
	printf("result: %f\n", idot(5, x, y));
}

int main(int argc, char *argv[])
{
	//test();
	int n = 10;
	int count = 10;

	NUM_THREADS = 64;
	omp_set_num_threads(NUM_THREADS);
	for (int i = 0; i < 7; ++i) {
		n *= 10;
		long double *x = (long double *) malloc(sizeof(long double) * n);
		long double *y = (long double *) malloc(sizeof(long double) * n);
		//long double abs_error = 0.0;
		//long double rel_error = 0.0;
		//double run_time = 0.0;

		// Input vector x and vector y.
		//double input_time = omp_get_wtime();
		intput(n, x, y);
		double *x_double = (double *) _mm_malloc(n * sizeof(double), 64);
		double *y_double = (double *) _mm_malloc(n * sizeof(double), 64);
#pragma omp parallel for num_threads(64)
		for (int i = 0; i < n; ++i) {
			x_double[i] = x[i];
			y_double[i] = y[i];
		}
		// Calculate the base value
		long double r = kahan_dot(n, x, y);

		//for (int k = 0; k < count; ++k) { // times of experiments
			// Calculate the output value
		double start_time = omp_get_wtime();
		double r_bar = idot(n, x_double, y_double);
		double run_time = omp_get_wtime() - start_time;

		// Absolute error
		long double abs_error = r > r_bar ? r - r_bar : r_bar - r;

		// Relative error
		long double rel_error = abs_error/r;
		//}
		//abs_error /= count;
		//rel_error /= count;
		//run_time /= count;

		printf("%d %.20Lf %.20Lf %f\n", n, abs_error, rel_error, run_time);

		_mm_free(x_double);
		_mm_free(y_double);
		free(x);
		free(y);
	}

	return 0;
}
