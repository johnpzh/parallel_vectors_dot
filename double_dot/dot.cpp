#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mkl.h>
#include <omp.h>

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
	for (int i = 0; i < n; ++i) {
		dot += x[i] * y[i];
	}

	return dot;
}
void test()
{
	int len = 10;
	double *x = (double *) malloc(sizeof(double) * len);
	double *y = (double *) malloc(sizeof(double) * len);

	srand(time(0));
	for (int i = 0; i < len; ++i) {
		x[i] = (double) rand()/RAND_MAX;
		y[i] = (double) rand()/RAND_MAX;
	}
	print_vector(x, len, "x");
	print_vector(y, len, "y");

	double r = cblas_ddot(len, x, 1, y, 1);
	double ir = iblas_dot(len, x, y);
	double dr = cblas_ddot(len, x, 1, y, 1);
	printf("r:  %.17f\n", r);
	printf("ir: %.17f\n", ir);
	printf("dr: %.17f\n", dr);
	printf("nr: %.17f\n", idot(len, x, y));

	float *x_s = (float *) malloc(sizeof(float) * len);
	float *y_s = (float *) malloc(sizeof(float) * len);
	for (int i = 0; i < len; ++i) {
		x_s[i] = x[i];
		y_s[i] = y[i];
	}

	printf("rs: %.17f\n", cblas_sdot(len, x_s, 1, y_s, 1));
	printf("is: %.17f\n", iblas_dot(len, x_s, y_s));
	printf("ds: %.17f\n", cblas_sdot(len, x_s, 1, y_s, 1));
	printf("ns: %.17f\n", idot(len, x_s, y_s));


	free(y_s);
	free(x_s);
	free(y);
	free(x);
}

void intput(int n, long double *x, long double *y)
{
	srand(time(0));
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
	//long double abs_errors[7];
	//long double rel_errors[7];
	//long double run_times[7];
	//memset(abs_errors, 0, sizeof(abs_errors));
	//memset(rel_errors, 0, sizeof(rel_errors));
	//memset(run_times, 0, sizeof(run_times));
	for (int i = 0; i < 7; ++i) {
		n *= 10;
		long double *x = (long double *) malloc(sizeof(long double) * n);
		long double *y = (long double *) malloc(sizeof(long double) * n);
		long double abs_error = 0.0;
		long double rel_error = 0.0;
		double run_time = 0.0;

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

		// Calculate the output value
		double start_time = omp_get_wtime();
		double r_bar = idot(n, x_double, y_double);
		run_time += omp_get_wtime() - start_time;

		// Absolute error
		abs_error += r > r_bar ? r - r_bar : r_bar - r;

		// Relative error
		rel_error += abs_error/r;

		printf("%d %.20Lf %.20Lf %f\n", n, abs_error, rel_error, run_time);

		free(x_double);
		free(y_double);
		free(x);
		free(y);
	}

	return 0;
}
