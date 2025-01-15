#ifndef NNLS_SCALER_H
#define NNLS_SCALER_H
#include "types.h"
#include "utils.h"
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
        const double thMin = 1.0e-14;
        const double thMax = 1.0e14;
        const double minSf = 1.0e-8;
        double scaleFactorSL = 1.0; // limited
        double scaleFactorSU = 1.0; // unlimited
        for (std::size_t i = 0; i < M.size(); ++i) {
            double norm2 = 0.0;
            for (std::size_t j = 0; j < M[i].size(); ++j) {
                norm2 += M[i][j] * M[i][j];
            }
            double s2 = s[i] * s[i];
            const double rat = norm2 / s2;
            if (thMin < rat && rat < thMax) {
                // check if ratio is in bounds in which scaling can be done
                // without possible numerical problems
                // Remark:
                // if rat is not small or big e.g. ~1, or e.g. ~1.0e-2
                // may be case when final scale factor will be unsufficient, but the
                // condition  thMin < rat && rat < thMax is satisfied not for all the constraints,
                // so other unbalanced constraints will be bad scaled
                const double ratRoot = sqrt(rat);
                double scaleFactor = ratRoot;
                scaleFactorSL = std::fmin(scaleFactor, scaleFactorSL);
            } else {
                double bf = 1.0; // balance factor
                if (rat <  thMin) {
                    bf = rat / thMin;  // < 1
                } else if (rat > thMax){
                    bf = rat / thMax;  // > 1
                }
                // s' = bf * s
                if (bf < scaleFactorSU) { // ||M|| << ||s||
                    scaleFactorSU = std::fmax(minSf, bf);
                }
            }
            //TODO: implement scaleFactor Up
            scaleCoefs[i] = norm2; // save norms
        }
        const bool blncL = isSame(scaleFactorSL, 1.0);
        const bool blncU = isSame(scaleFactorSU, 1.0);
        if (blncL && !blncU) {
            // all the constraints are very unbalanced
            scaleFactorS = scaleFactorSU;

        } else if (!blncL && blncU) {
            // all the constraints are good balanced
            scaleFactorS = scaleFactorSL;
        } else if (blncL && blncU) {
            // all the constraints are very good balanced
            scaleFactorS = 1.0;
        } else {
            // exist good and bad balanced constraints
            // balance only good constraints
            scaleFactorS = scaleFactorSL;
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
