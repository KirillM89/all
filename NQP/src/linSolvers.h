#ifndef LINSOLVERS_H
#define LINSOLVERS_H
#include <unordered_map>
#include "types.h"
namespace QP_NNLS {
class ILinSolver {
    // Interface for linear solver
    // [M s] * [M_T s_T] * y = - gamma * s
    // methods Add() and Delete() calls when corresponding constraint
    // adds/deletes to/from active set
    // Solve() calls in place where the problem has to be solved
public:
    virtual ~ILinSolver() = default;
    virtual bool Add(const std::vector<double>& mp, double sp, unsg_t indx) = 0;
    virtual bool Delete(unsg_t indx) = 0;
    virtual void SetGamma(double gamma) = 0;
    virtual const LinSolverOutput& Solve() = 0;
protected:
    ILinSolver() = default;
};

class CumulativeSolver: public ILinSolver {
    // Add / Delete methods constructs linear system
    // Solve() solves pre-constructed linear system
public:
    CumulativeSolver() = delete;
    CumulativeSolver(const matrix_t& M, const std::vector<double>& s);
    virtual ~CumulativeSolver() override = default;
    virtual bool Add(const std::vector<double>& mp, double sp, unsg_t indx) override;
    virtual bool Delete(unsg_t indx) override;
    virtual void SetGamma(double gamma) override {this->gamma = gamma;}
protected:
    const unsg_t nConstraints;
    unsg_t nVariables;
    unsg_t nActive;
    double gamma;
    std::vector<bool> activeSet;
    const matrix_t& M;
    const std::vector<double>& s;

};

class CumulativeLDLTSolver: public CumulativeSolver {
public:
    CumulativeLDLTSolver() = delete;
    CumulativeLDLTSolver(const matrix_t& M, const std::vector<double>& s);
    virtual ~CumulativeLDLTSolver() override = default;
    const LinSolverOutput& Solve() override;
protected:
    LinSolverOutput output;
};

class DynamicSolver : public ILinSolver {
    // Solver based on dynamically updated LDLT decomposition
    // Add / Delete methods recompute LDL
    // Solve() solves LDLT * x = b with already computed L and D

};
}
#endif // LINSOLVERS_H
