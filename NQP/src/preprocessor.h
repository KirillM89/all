#ifndef QP_NNLS_PREPROCESSOR_H
#define QP_NNLS_PREPROCESSOR_H
#include "types.h"
namespace QP_NNLS {
class NqpPreprocessor {
public:
    NqpPreprocessor();
    virtual ~NqpPreprocessor() = default;
    void Prepare(const DenseQPProblem& problem);
    const DenseQPProblem& GetProblem() const;
    void RecomputeQutput(SolverOutput& sol) const;
private:
    const DenseQPProblem* origProblem = nullptr;
    DenseQPProblem newProblem;
    std::vector<double> hDiag;
    matrix_t mPmt;
    matrix_t PlInv;
};
} // QP_NNLS
#endif // QP_NNLS_PREPROCESSOR_H
