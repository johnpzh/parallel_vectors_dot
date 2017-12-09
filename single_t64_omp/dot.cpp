#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mkl.h>
#include <omp.h>

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

template<typename Float>
Float idot(int n, Float *x, Float *y)
{
	Float dot = 0.0;
#pragma omp parallel for reduction(+: dot)
	for (int i = 0; i < n; ++i) {
		dot += x[i] * y[i];
	}

	return dot;
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
		long double abs_error = 0.0;
		long double rel_error = 0.0;
		double run_time = 0.0;

		// Input vector x and vector y.
		intput(n, x, y);
		float *x_single = (float *) malloc(n * sizeof(float));
		float *y_single = (float *) malloc(n * sizeof(float));
#pragma omp parallel for num_threads(64)
		for (int i = 0; i < n; ++i) {
			x_single[i] = x[i];
			y_single[i] = y[i];
		}

		// Calculate the base value
		long double r = kahan_dot(n, x, y);

		// Calculate the output value
		double start_time = omp_get_wtime();
		float r_bar = idot(n, x_single, y_single);
		run_time += omp_get_wtime() - start_time;

		// Absolute error
		abs_error += r > r_bar ? r - r_bar : r_bar - r;

		// Relative error
		rel_error += abs_error/r;

		printf("%d %.20Lf %.20Lf %f\n", n, abs_error, rel_error, run_time);

		free(x_single);
		free(y_single);
		free(x);
		free(y);
	}

	return 0;
}
