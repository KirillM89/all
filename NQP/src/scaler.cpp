#include <cmath>
#include "scaler.h"
#include "utils.h"
namespace QP_NNLS {
MBScaler::MBScaler(matrix_t& A, matrix_t& M, std::vector<double>& b):
    A(A), M(M), b(b)
{}

void MBScaler::Scale() {
    const std::size_t nc = b.size();
    const std::size_t nv = M.front().size();
    scaleCoefs.resize(nc, 1.0);
    for (std::size_t c = 0; c < nc; ++c) {
         double maxRowComponent = 0.0;
        for (std::size_t v = 0; v < nv; ++v) {
            maxRowComponent = std::fmax(std::fabs(M[c][v]), maxRowComponent);
        }
        if (maxRowComponent > 0.0) {
            const double scaleFactor = 1.0 / maxRowComponent;
            for (std::size_t v = 0; v < nv; ++v) {
                M[c][v] *= scaleFactor;
                A[c][v] *= scaleFactor;
            }
            b[c] *= scaleFactor;
            scaleCoefs[c] = scaleFactor;
        }
    }
}

}
