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
    CumulativeSolver();
    ~CumulativeSolver() override = default;
    bool Add(const std::vector<double>& mp, double sp, unsg_t indx) override;
    bool Delete(unsg_t indx) override;
    void SetGamma(double gamma) override {this->gamma = gamma;}
protected:
    double gamma;
    std::list<std::vector<double>> mA;
    std::list<double> vB;
    std::unordered_map<unsg_t, std::list<std::vector<double>>::iterator> indicesMA;
    std::unordered_map<unsg_t, std::list<double>::iterator> indicesVB;
};

class CumulativeLDLTSolver: public CumulativeSolver {
public:
    CumulativeLDLTSolver() = default;
    ~CumulativeLDLTSolver() override = default;
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
