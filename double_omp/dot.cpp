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

void test()
{
	int n = 10E8;
	printf("n: %d\n", n);
	exit(1);
}

int main(int argc, char *argv[])
{
	//test();
	//int n = 10E8;
	int n = 1E8;
	long double *x = (long double *) malloc(sizeof(long double) * n);
	long double *y = (long double *) malloc(sizeof(long double) * n);
	// Input vector x and vector y.
	intput(n, x, y);
	double *x_double = (double *) malloc(n * sizeof(double));
	double *y_double = (double *) malloc(n * sizeof(double));
#pragma omp parallel for num_threads(64)
	for (int i = 0; i < n; ++i) {
		x_double[i] = x[i];
		y_double[i] = y[i];
	}

	// Calculate the base value
	long double r = kahan_dot(n, x, y);

	for (int i = 1; i < 65; i *= 2) {
		NUM_THREADS = i;
		omp_set_num_threads(NUM_THREADS);
	//for (int i = 0; i < 7; ++i) {
	
		long double abs_error = 0.0;
		long double rel_error = 0.0;
		double run_time = 0.0;

		//for (int k = 0; k < count; ++k) { // times of experiments
			// Calculate the output value
		double start_time = omp_get_wtime();
		double r_bar = idot(n, x_double, y_double);
		run_time += omp_get_wtime() - start_time;

		// Absolute error
		abs_error += r > r_bar ? r - r_bar : r_bar - r;

		// Relative error
		rel_error += abs_error/r;
		//}
		//abs_error /= count;
		//rel_error /= count;
		//run_time /= count;

		printf("%d %.20Lf %.20Lf %f\n", NUM_THREADS, abs_error, rel_error, run_time);

	//}
	}
	free(x_double);
	free(y_double);
	free(x);
	free(y);

	return 0;
}
