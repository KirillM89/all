#ifndef QP_NNLS_CALLBACK_H
#define QP_NNLS_CALLBACK_H
#include <memory>
#include "types.h"
#include "log.h"
namespace QP_NNLS {
    struct IterationData {
       std::deque<unsg_t>* activeSetHistory;
       std::vector<double>* dual;
       std::vector<double>* primal;
       std::vector<double>* violations;
       std::vector<double>* zp;
       std::set<unsg_t>* activeSet;
       double gamma;
       double dualTol;
       double rsNorm;
       unsg_t newIndex;
       unsg_t iteration;
       bool singular;
    };
    struct FinalData {
        PrimalLoopExitStatus primalStatus;
        DualLoopExitStatus dualStatus;
        unsg_t nIterations;
        double cost;
        std::vector<double>* violations;
        std::vector<double>* x;
        std::vector<double>* lambda;
        std::vector<double>* lambdaUp;
        std::vector<double>* lambdaLw;
        std::vector<LinSolverTime>* linSlvrTimes;
    };
    struct InitializationData {
        unsg_t nVariables;
        unsg_t nConstraints;
        unsg_t nEqConstraints;
        double scaleDB;
        std::string tChol;
        std::string tInv;
        std::string tM;
        std::vector<double>* s;
        std::vector<double>* b;
        std::vector<double>* c;
        matrix_t* Chol;
        matrix_t* CholInv;
        matrix_t* M;
        InitStageStatus InitStatus;
    };
    class Callback {
    public:
        Callback() = default;
        virtual ~Callback() = default;
        virtual void Init() {
            return;
        }
        virtual void SetLogLevel(unsg_t logLevel) {
            return;
        }
        virtual void ProcessData(int stage) {
            return;
        };
        InitializationData initData;
        IterationData iterData;
        FinalData finalData;
    };

    class Callback1 : public Callback {
    public:
        Callback1(const std::string& filePath);
        virtual ~Callback1() override = default;
        void ProcessData(int stage) override;
        void SetLogLevel(unsg_t logLevel) override {
            this -> logLevel = logLevel;
        }
        void Init() override;
    private:
        const std::string filePath;
        std::unique_ptr<Logger> logger;
        unsg_t logLevel = 0u;
    };
}

#endif // CALLBACK_H
