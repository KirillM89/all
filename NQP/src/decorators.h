#ifndef QP_NNLS_DECORATORS_H
#define QP_NNLS_DECORATORS_H
#include <memory>
#include "types.h"
#include "callback.h"
namespace QP_NNLS {
    class Core;
    class QPNNLS {
    public:
        void Init(const Settings& settings);
        void SetCallback(std::unique_ptr<Callback> callback);
        const SolverOutput& GetOutput();
    protected:
        QPNNLS();
         ~QPNNLS();
        QPNNLS(const QPNNLS& other) = delete;
        QPNNLS(QPNNLS&& other) = delete;
        QPNNLS& operator=(const QPNNLS& other) = delete;
        QPNNLS& operator=(QPNNLS&& other) = delete;
        bool VerifySettings(const Settings& settings);
        SolverOutput output;
        std::unique_ptr<Core> core;
        bool isInitialized = false;
    };

    class QPNNLSDense : public QPNNLS {
    public:
        bool SetProblem(const DenseQPProblem& problem);
        void Solve();
    };

    class QPNNLSSparse : public QPNNLS {
    public:
        void Solve(const SparseQPProblem& problem);
    };
}

#endif // DECORATORS_H
