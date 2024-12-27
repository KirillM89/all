#ifndef QP_QLD_H
#define QP_QLD_H
namespace QP_SOLVERS {
int ql0001_(const int * m, const int * me, const int * mmax, const int * n, const int * nmax, const int * mnn,
            double * c, const double * d, const double * a, const double * b, const double * xl,
            const double * xu, double * x, double * u, const int * iout, int * ifail,
            const int * log_level, double * war, int * lwar, int * iwar, int * liwar,
            const double * eps1); 
}
#endif