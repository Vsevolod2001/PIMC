#include "mpi.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <gsl/gsl_sf_bessel.h>

#include "code/nr3.h"
#include "code/ran.h"

using namespace std;

Ran *ran;

const int d = 1, N = 128;
int Ns = 2048,  Na = 8, Np = 36;
double beta = 16, m = 100, omega = 1, tau = 1.0 * beta / N;
double q[N][d];

void InitCold(double q[][d]) {
    for (int site = 0; site < N; site++)
        for (int dimension = 0; dimension < d; dimension++)
            q[site][dimension] = 0;
}

double U(double x0, double d) {
    return x0 + (1.0 * ran->doub() * 2 - 1) * d;
}

void InitWarm(double q[][d], int N, int d, int rank, int size, int path){
    for (int site = 0; site < N; site++)
        for (int dimension = 0; dimension < d; dimension++)
            q[site][dimension] = U(tau, pow(tau, 0.5));
}

double V(double Q) {
    return 1.0 / 2 * m * omega * omega * Q * Q;
}

double P(double q1, double q2, double Q) {
    double arg1 = 1.0 * pow(tau * tau + q1, 0.5), arg2 = 1.0 * pow(tau * tau + q2, 0.5);
    return (
        1.0 / pow(arg1 * arg2, 1.0 * (d + 1) / 2) *
        gsl_sf_bessel_Knu(1.0 * (d + 1) / 2, 1.0 * m * arg1) *
        gsl_sf_bessel_Knu(1.0 * (d + 1) / 2, 1.0 * m * arg2) *
        exp(-1.0 * tau * V(pow(Q, 0.5)))
    );
}

double GetP(int prev, int curr, int next, double *dq) {
    double prev_curr = 0.0, curr_next = 0.0, prev_Q = 0.0, Q_next = 0.0, Q_curr = 0.0, Q = 0.0;
    for (int i = 0; i < d; i++) {
        prev_curr += 1.0 * pow(q[prev][i] - q[curr][i], 2);
        curr_next += 1.0 * pow(q[curr][i] - q[next][i], 2);
        prev_Q += 1.0 * pow(q[prev][i] - q[curr][i] - dq[i], 2);
        Q_next += 1.0 * pow(q[curr][i] + dq[i] - q[next][i], 2);
        Q_curr += 1.0 * pow(q[curr][i], 2);
        Q += 1.0 * pow(q[curr][i] + dq[i], 2);
    }
    return 1.0 * P(prev_Q, Q_next, Q) / P(prev_curr, curr_next, Q_curr);
}

void Sweep() {
    double dq[d];
    for (int site = 0; site < N; site++) {
        int curr = rand() % N, prev = N - 1, next = 0;
        if (curr != 0) prev = curr - 1;
        if (curr != (N - 1)) next = curr + 1;
        for (int attempt = 0; attempt < Na; attempt++) {
            for (int dimension = 0; dimension < d; dimension++) dq[dimension] = 1.0 * U(0, pow(tau, 0.5)); // tau
            double p = 1.0 * GetP(prev, curr, next, dq);
            if ((ran -> doub()) < p) for (int dimension = 0; dimension < d; dimension++) q[curr][dimension] += 1.0 * dq[dimension];
        }
    }
}

double GetX2() {
  double result = 0;
  for (int site = 0; site < N; site++)
    for (int dimension = 0; dimension < d; dimension++)
      result += 1.0 * q[site][dimension] * q[site][dimension];
  return 1.0 * result / N;
}

double GetV() {
    double result = 0.0;
    for (int site = 0; site < N; site++) {
        double point = 0;
        for (int dimension = 0; dimension < d; dimension++) point += 1.0 * q[site][dimension] * q[site][dimension];
        result += 1.0 * V(pow(point, 0.5));
    }
  return 1.0 * result / N;
}

double GetK() {
    double result = 0.0;
    for (int site = 0; site < N; site++) {
        int next = 0;
        if (site < (N - 1)) next = site + 1;
        double dq2 = 0, tau2 = tau * tau;
        for (int dimension = 0; dimension < d; dimension++) dq2 += 1.0 * pow(q[site][dimension] - q[next][dimension], 2);
        double arg = 1.0 * m * pow(tau2 + dq2, 0.5);
        result += 1.0 * m * m * tau / arg * gsl_sf_bessel_Knu(1.0 * (d + 3) / 2, 1.0 * arg) / gsl_sf_bessel_Knu(1.0 * (d + 1) / 2, 1.0 * arg) - 1.0 / tau - m;
    }
    return 1.0 * result / N;
}

void GetCorr(double corr[][N], int number) {
    for (int site = 0; site < N; site++) corr[number][site] = 0.0;
    for (int shift = 0; shift < N; shift++)
        for (int site = 0; site < N; site++)
            for (int dimension = 0; dimension < d; dimension++)
                corr[number][shift] += 1.0 * q[site][dimension] * q[(site + shift) % N][dimension];
}

void MeanCorr(double corr[][N], double *mean_corr, int size) {
    for (int site = 0; site < N; site++) mean_corr[site] = 0;
    for (int site = 0; site < N; site++)
        for (int number = 0; number < size; number++)
            mean_corr[site] += corr[number][site];
    for (int site = 0; site < N; site++) mean_corr[site] = 1.0 * mean_corr[site] / size / N;
}

double Mean(double *a, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) result += 1.0 * a[i];
    return 1.0 * result / n;
}

double Variance(double *a, int n) {
    double mean = Mean(a, n), result = 0;
    for (int i = 0; i < n; i++) result += pow(mean - a[i], 2);
    return pow(result / n, 0.5); // n - 1
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    timeval t;
    gettimeofday(&t, NULL);
    int seed = int(t.tv_sec + t.tv_usec);
    ran = new Ran(seed);
    srand(seed);

    double x2[Np / size  + 1], v[Np / size  + 1], k[Np / size  + 1], corr[Np / size + 1][N], mean_corr[N];
    int x2_size[size], v_size[size], k_size[size], corr_size[size];
    for (int number = 0; number < Np / size + 1; number++) {
        x2_size[number] = 0;
        v_size[number] = 0;
        k_size[number] = 0;
        corr_size[number] = 0;
    }

    for (int path = 0; path < Np; path++) {
        if (path % size == rank) {
            printf("path %d of %d\n", path + 1, Np);
            InitCold(q);
            for (int sweep = 0; sweep < Ns; sweep++) Sweep();
            x2[x2_size[rank]] = GetX2();
            x2_size[rank] += 1;

            v[v_size[rank]] = GetV();
            v_size[rank] += 1;

            k[k_size[rank]] = GetK();
            k_size[rank] += 1;

            GetCorr(corr, corr_size[rank]);
            corr_size[rank] += 1;
        }
    }

    printf("%d %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf ",
        Ns,
        Mean(x2, x2_size[rank]), Variance(x2, x2_size[rank]),
        Mean(v, v_size[rank]), Variance(v, v_size[rank]),
        Mean(k, k_size[rank]), Variance(k, k_size[rank])
    );
    MeanCorr(corr, mean_corr, corr_size[rank]);
    for (int site = 0; site < N; site ++) printf("%.10lf ", mean_corr[site]);
    printf("%.10lf \n", mean_corr[0]);
    FILE *output = fopen("output.dat", "a");
    fprintf(output, "%d %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf ",
        Ns,
        Mean(x2, x2_size[rank]), Variance(x2, x2_size[rank]),
        Mean(v, v_size[rank]), Variance(v, v_size[rank]),
        Mean(k, k_size[rank]), Variance(k, k_size[rank])
    );
    for (int site = 0; site < N; site ++) fprintf(output, "%.10lf ", mean_corr[site]);
    fprintf(output, "%.10lf \n", mean_corr[0]);
    fclose(output);
    MPI_Finalize();
    return 0;
}