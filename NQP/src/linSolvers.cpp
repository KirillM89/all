#include "linSolvers.h"
#include "NNLSQPSolver.h" // MMTBSolver
namespace QP_NNLS {
CumulativeSolver::CumulativeSolver():
    gamma(1.0),
    mA{},
    vB{},
    indicesMA{},
    indicesVB{}
{}
bool CumulativeSolver::Add(const std::vector<double>& mp, double sp, unsg_t indx) {
    std::vector<double> vAdd(mp.size() + 1);
    vAdd.back() = sp;
    indicesMA[indx] = mA.insert(mA.end(), vAdd);
    indicesVB[indx] = vB.insert(vB.end(), -sp);
}
bool CumulativeSolver::Delete(unsg_t indx) {
    mA.erase(indicesMA[indx]);
    vB.erase(indicesVB[indx]);
    indicesMA.erase(indx);
    indicesVB.erase(indx);
}
const LinSolverOutput& CumulativeLDLTSolver::Solve() {
    const matrix_t M(mA.begin(), mA.end());
    const std::vector<double> b(vB.begin(), vB.end());
    MMTbSolver mmtb;
    mmtb.Solve(M, b);
    return output;
}



}
