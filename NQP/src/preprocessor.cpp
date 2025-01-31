#include "preprocessor.h"
#include "utils.h"
#include <iostream>
namespace QP_NNLS {
NqpPreprocessor::NqpPreprocessor():
    origProblem(nullptr)
{}
void NqpPreprocessor::Prepare(const DenseQPProblem& problem) {
    origProblem = &problem;
    newProblem = problem;
    const std::size_t nv = problem.H.size();
    const std::size_t nc = problem.A.size();
    mPmt.resize(nv, std::vector<double>(nv));
    std::size_t izr, ing;
    InPlaceLdlt(newProblem.H, mPmt, izr, ing);
    matrix_t L(newProblem.H);
    for (std::size_t i = 0; i < nv; ++i) {
        L[i][i] = 1.0;
        std::cout << newProblem.H[i][i] << " ";
        for (std::size_t j = i + 1 ; j < nv; ++j) {
            L[i][j] = 0.0;
            //problem.H[nv - 1 - i][nv - 1 - j] = 0.0;
            newProblem.H[i][j] = 0.0;
            newProblem.H[j][i] = 0.0;
        }
    }
    PlInv.resize(nv, std::vector<double>(nv));
    matrix_t Linv(PlInv);
    InvertCholetsky(L, Linv);
    M1M2T(mPmt, Linv, PlInv);
    MultTransp(PlInv, problem.c, newProblem.c);
    Mult(problem.A, PlInv, newProblem.A);
    //extend A
    for (std::size_t i = 0; i < nv; ++i) {
        newProblem.A.push_back(PlInv[i]);
        newProblem.A.push_back(-PlInv[i]);
        newProblem.b.push_back(problem.up[i]);
        newProblem.b.push_back(-problem.lw[i]);
        newProblem.up[i] = 1.0e10;
        newProblem.lw[i] = -1.0e10;
    }
}

const DenseQPProblem& NqpPreprocessor::GetProblem() const {
    return newProblem;
}

void NqpPreprocessor::RecomputeQutput(SolverOutput& sol) const {
    sol.nConstraints -= 2 * sol.nVariables; // include bounds
    std::size_t nc = sol.nConstraints - 2 * sol.nVariables;
    //recompute Lambda
    for (std::size_t i = 0; i < sol.nVariables; ++i) {
        sol.lambdaUp[i] = sol.lambda[nc + 2 * i];
        sol.lambdaLw[i] = sol.lambda[nc + 2 * i + 1];
    }
    sol.lambda.resize(sol.nConstraints);
    std::vector<double> x = sol.x;
    Mult(PlInv, x, sol.x);
}

} // QP_NNLS
