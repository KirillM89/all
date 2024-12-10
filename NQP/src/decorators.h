#ifndef QP_NNLS_DECORATORS_H
#define QP_NNLS_DECORATORS_H
#include <memory>
#include <vector>
#include <unordered_set>
#include "types.h"
namespace QP_NNLS {
    struct IterationData {
       std::vector<double>* dual;
       std::vector<double>* primal;
       std::vector<double>* violations;
       std::unordered_set<unsg_t>* activeSet;
       double gamma;
       unsg_t newIndex;
       unsg_t iteration;
       bool singular;
    };

    struct FinalData {
        PrimalLoopExitStatus primalStatus;
        DualLoopExitStatus dualStatus;
        unsg_t nIterations;
        double cost;
        std::vector<double> violations;
        std::vector<double> x;
        std::vector<double> lambda;
        std::vector<double> lambdaUp;
        std::vector<double> lambdaLw;
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
        virtual void ProcessData(int stage) {
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
        void SetCallback(std::unique_ptr<Callback> callback);
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
        void SetProblem(const DenseQPProblem& problem);
        void Solve();
    };

    class QPNNLSSparse : public QPNNLS {
    public:
        void Solve(const SparseQPProblem& problem);
    };

}

#endif // DECORATORS_H
