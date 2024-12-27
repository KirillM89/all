#ifndef NNLS_SCALER_H
#define NNLS_SCALER_H
#include "types.h"
#include <cmath>
namespace QP_NNLS {

class MBScaler {
public:
    MBScaler() = delete;
    MBScaler(matrix_t& A, matrix_t& M, std::vector<double>& b);
    ~MBScaler() = default;
    void Scale();
    const std::vector<double>& GetScaleCoefs() {
        return scaleCoefs;
    }
private:
    matrix_t& A;
    matrix_t& M;
    std::vector<double>& b;
    std::vector<double> scaleCoefs;
    const double maxMValue = 1.0e7;
};

struct ScaleCoefs {
    double scaleFactorS;
};

class OrtScaler {
public:
    OrtScaler() = delete;
    OrtScaler(matrix_t& M, std::vector<double>& s):
        M(M), s(s)
    {}
    ~OrtScaler() = default;

    void Scale() {
        scaleCoefs.resize(M.size());
        balanceFactor.resize(M.size(), 1.0);
        const double thMin = 1.0e-10;
        const double thMax = 1.0e10;
        const double minSf = 1.0e-7;
        scaleFactorS = 1.0;
        for (std::size_t i = 0; i < M.size(); ++i) {
            double norm2 = 0.0;
            for (std::size_t j = 0; j < M[i].size(); ++j) {
                norm2 += M[i][j] * M[i][j];
            }
            double s2 = s[i] * s[i];
            const double rat = norm2 / s2;
            double bf = 1.0;
            if (rat <  thMin) {
                bf = rat / thMin;  // < 1
            } else if (rat > thMax){
                bf = rat / thMax;  // > 1
            }
            // s' = bf * s
            if (bf < scaleFactorS) { // ||M|| << ||s||
                scaleFactorS = std::fmax(minSf, bf);
            }
            //TODO: implement scaleFactor Up
            scaleCoefs[i] = norm2; // save norms
        }

        for (std::size_t i = 0; i < M.size(); ++i) {
            s[i] *= scaleFactorS;
            scaleCoefs[i] = 1.0 / sqrt(scaleCoefs[i] + s[i] * s[i]);
            s[i] *= scaleCoefs[i];
            for (std::size_t j = 0; j < M[i].size(); ++j) {
                M[i][j] *= scaleCoefs[i];
            }
        }
        sCoefs.scaleFactorS = scaleFactorS;
    }
    void UnScale(std::vector<double>& lambda) {
        for (std::size_t i = 0; i < lambda.size(); ++i) {
            lambda[i] *= scaleCoefs[i];
        }
    }
    const ScaleCoefs& GetScaleCoefs() {
        return sCoefs;
    }

private:
    matrix_t& M;
    std::vector<double>& s;
    double scaleFactorS = 1.0;
    std::vector<double> scaleCoefs;
    std::vector<double> balanceFactor;
    ScaleCoefs sCoefs;
};
}
#endif // NNLS_SCALER_H
