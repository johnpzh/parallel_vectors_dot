#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mkl.h>

void print_vector(double *x, int len, char *name = "vector")
{
	printf("%s:", name);
	for (int i = 0; i < len; ++i) {
		printf(" %f", x[i]);
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
}

void single_dot(int n, long double *x, long double *y)
{
}

void double_dot(int n, long double *x, long double *y)
{
}

int main(int argc, char *argv[])
{
	test();
	long double *x = (long double *) malloc(sizeof(long double) * len);
	long double *y = (long double *) malloc(sizeof(long double) * len);

	// Input vector x and vector y.

	// Caldulate

	free(x);
	free(y);

	return 0;
}
