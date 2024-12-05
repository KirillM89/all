#ifndef QP_NNLS_DECORATORS_H
#define QP_NNLS_DECORATORS_H
#include <memory>
#include <vector>
#include "types.h"
namespace QP_NNLS {
    struct IterationData {

    };

    struct FinalData {

    };

    struct InitializationData {
        double dbSacleFactor;
        std::string tChol;
        std::string tInv;
        std::string tM;
        std::vector<double> s;
        std::vector<double> b;
        std::vector<double> c;
        matrix_t Chol;
        matrix_t CholInv;
        matrix_t M;
        InitStageStatus InitStatus;
    };

    class Callback {
    public:
        Callback() = default;
        virtual ~Callback() = default;
        virtual void ProcessData() {
            return;
        };
        InitializationData initData;
        IterationData iterData;
        FinalData finalData;
    };

    class Core;
    class QPNNLS {
    public:
        void Init(const Settings& settings);
        void SetCallback(std::shared_ptr<Callback> callback);
        const SolverOutput& GetOutput();
    protected:
        QPNNLS();
         ~QPNNLS() = default;
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
        void Solve(const DenseQPProblem& problem);
    };

    class QPNNLSSparse : public QPNNLS {
    public:
        void Solve(const SparseQPProblem& problem);
    };

}

#endif // DECORATORS_H
