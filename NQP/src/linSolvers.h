#ifndef LINSOLVERS_H
#define LINSOLVERS_H
#include "types.h"
namespace QP_NNLS {
class iLinSolver {
    // interface for linear solver
    // [M s] * [M_T s_T] * y = - gamma * s
public:
    virtual ~iLinSolver() = default;
    bool Add(const std::vector<double>& mp, double sp, unsg_t indx) = 0;
    bool Delete(unsg_t indx) = 0;
    void SetGamma(double gamma) = 0;
    void Solve() = 0;
protected:
    iLinSolver() = default;
};

class CumulativeSolver: public iLinSolver {

};

class DynamicSolver : public iLinSolver {

};
}
#endif // LINSOLVERS_H
