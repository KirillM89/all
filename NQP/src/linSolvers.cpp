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
bool CumulativeSolver::Add(unsg_t indx) {
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
    gamma(1.0), ldlt(M, s), ndzero(0), maxSize(s.size()), S(s),
    forward(std::vector<double>(maxSize)),
    backward(std::vector<double>(maxSize))
{ }

bool CumulativeLDLTSolver::Add(unsg_t indx) {
    ldlt.Add(indx);
    return true;
}
bool CumulativeLDLTSolver::Delete(unsg_t indx) {
    ldlt.Delete(indx);
    return true;
}
const LinSolverOutput& CumulativeLDLTSolver::Solve() {
    const matrix_t& l = ldlt.GetL();
    const std::vector<double>& d = ldlt.GetD();
    const std::list<unsigned int>& rows = ldlt.GetRows();
    const std::size_t nr = rows.size();
    if (nr == 0) {
        output.solution = std::vector<double>(maxSize, 0.0);
        output.indices.clear();
    } else {
        std::size_t i = 0;
        for (auto iAct : rows) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += l[i][j] * forward[j];
            }
            forward[i] = gamma * S[iAct] - sum;
            ++i;
        }
        for (int i = nr - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < nr; ++j) {
                sum += l[j][i] * d[i] * backward[j];
            }
            backward[i] = (std::fabs(d[i]) < zeroTol) ? 0.0 : (forward[i] - sum) / d[i];
        }
        output.solution = backward;
        output.indices = rows;
        output.nDNegative = 0;
    }
    return output;


    /*
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
    */
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
