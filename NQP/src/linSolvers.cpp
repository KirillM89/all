#include "linSolvers.h"
#include "utils.h"
namespace QP_NNLS {
CumulativeSolver::CumulativeSolver(const matrix_t& M,
                                   const std::vector<double>& s ):
    nConstraints(M.size()),
    nVariables(0),
    nActive(0),
    gamma(1.0),
    M(M),
    s(s)
{
    if (nConstraints > 0) {
        nVariables = M.front().size();
    }
    activeSet.resize(nConstraints, false);
}
bool CumulativeSolver::Add(const std::vector<double>& mp, double sp, unsg_t indx) {
    activeSet[indx] = true;
    ++nActive;
    return true;
}
bool CumulativeSolver::Delete(unsg_t indx) {
    if (activeSet[indx]) {
        activeSet[indx] = false;
        if (nActive > 0) {
            --nActive;
        }
    }
    return true;
}
CumulativeLDLTSolver::CumulativeLDLTSolver(const matrix_t& M,
                                           const std::vector<double>& s):
    CumulativeSolver(M, s), solver(M, s)
{}

const LinSolverOutput& CumulativeLDLTSolver::Solve() {
    /*
    matrix_t m;
    std::vector<double> b;
    std::vector<double> vAdd;
    output.indices.clear();
    for (unsg_t i = 0; i < nConstraints; ++i) {
        if (activeSet[i]) {
            vAdd = M[i];
            vAdd.push_back(s[i]);
            m.push_back(vAdd);
            b.push_back(-gamma * s[i]);
            output.indices.push_back(i);
        }
    }
    */
    std::set<unsigned int> active;
    output.indices.clear();
    for (unsg_t i = 0; i < nConstraints; ++i) {
        if (activeSet[i]) {
            active.insert(i);
            output.indices.push_back(i);
        }
    }
    if (!active.empty()) {
        output.nDNegative = solver.Solve(active, -gamma);
        output.solution = solver.GetSolution();
    } else {
        output.solution = std::vector<double>(nConstraints, 0.0);
    }
    return output;
}

CumulativeEGNSolver::CumulativeEGNSolver(const matrix_t& M,
                                         const std::vector<double>& s):
    CumulativeSolver(M, s)
{}

const LinSolverOutput& CumulativeEGNSolver::Solve() {
    output.indices.clear();
    if (nActive  == 0) {
        output.solution =  std::vector<double>(nConstraints, 0.0);
    } else {
        Eigen::MatrixXd A(nActive, nActive);
        Eigen::VectorXd b(nActive);
        output.solution = std::vector<double>(nActive, 0.0);
        unsg_t ii = 0;
        for (unsg_t i = 0; i < nConstraints; ++i) {
            if (activeSet[i]) {
                output.indices.push_back(i);
                unsg_t jj = 0;
                for (unsg_t c = 0; c < nConstraints; ++c) {
                    if (activeSet[c]) {
                        A(ii, jj) = 0.0;
                        for (unsg_t j = 0; j < nVariables; ++j) {
                            A(ii, jj) += (M[i][j] * M[c][j]);
                        }
                        A(ii, jj) += s[i] * s[c];
                        ++jj;
                    }
                }
                b(ii) = -gamma * s[i];
                ++ii;
            }
        }
        SolveByEGN(A, b);
    }
    return output;
}

void CumulativeEGNSolver::SolveByEGN(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::VectorXd r = A.ldlt().solve(b);
    for (unsg_t i = 0; i < nActive; ++i) {
        output.solution[i] = r[i];
    }
}
MssCumulativeSolver::MssCumulativeSolver(const matrix_t& M,
                                         const std::vector<double>& s):
    CumulativeSolver(M, s)
{}

const LinSolverOutput& MssCumulativeSolver::Solve() {
    output.indices.clear();
    if (nActive  == 0) {
        output.solution =  std::vector<double>(nConstraints, 0.0);
    } else {
        Eigen::MatrixXd A(nVariables + 1, nActive);
        Eigen::VectorXd b(nVariables + 1);
        output.solution = std::vector<double>(nActive, 0.0);
        for (unsg_t r = 0; r < nVariables + 1; ++r) {
            b(r) = 0.0;
            unsg_t act = 0;
            for (unsg_t c = 0; c < nConstraints; ++c) {
                if (activeSet[c]) {
                    if (r == 0) {
                        output.indices.push_back(c);
                    }
                    if (r == nVariables) {
                        A(r, act) = s[c];
                    } else {
                        A(r, act) = M[c][r];
                    }
                    ++act;
                }
            }
        }
        b(nVariables) = -gamma;
        SolveByEGN(A, b);
    }
    return output;
}

void MssCumulativeSolver::SolveByEGN(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::VectorXd r = A.colPivHouseholderQr().solve(b);
    for (unsg_t i = 0; i < nActive; ++i) {
        output.solution[i] = r[i];
    }
}

} //namespace QP_NNLS
