#ifndef NNLS_CORE_H
#define NNLS_CORE_H
#include <memory>
#include <map>
#include "types.h"
#include "linSolvers.h"
#include "scaler.h"
#include "timers.h"
#include "callback.h"
namespace QP_NNLS {
class Core {
    struct WorkSpace {
        WorkSpace() {
            Clear();
        }
        std::vector<double> s;
        std::vector<double> zp;
        std::vector<double> primal;
        std::vector<double> dual;
        std::vector<double> lambda;
        std::vector<double> MTY;
        std::vector<double> x;
        std::vector<double> c;
        std::vector<double> b;
        std::vector<double> v;
        std::vector<double> slack;
        std::vector<double> violations;
        std::vector<int> pmt;
        std::set<unsigned int> activeConstraints;
        std::set<unsigned int> linEqConstraints;
        std::unordered_set<unsigned int> negativeZp;
        matrix_t H;
        matrix_t M;
        matrix_t Jac;
        matrix_t Chol;
        matrix_t CholInv;
        matrix_t MS;
        std::deque<unsg_t> addHistory;
        void Clear();
    };

    enum PrimalRetStatus {
        SINGULARITY = 0x01,
        LINE_SEARCH_FAILED = 0x02
    };

public:
    Core();
    ~Core() = default;
    void Set(const CoreSettings& settings);
    void ResetProblem();
    void SetCallback(std::unique_ptr<Callback> callback);
    bool InitProblem(const DenseQPProblem& problem);
    void Solve();
    const SolverOutput& GetOutput() { return output; }
    InitStageStatus GetInitStatus() { return initStatus; }
private:
    DenseQPProblem* problem = nullptr;
    unsg_t nVariables;
    unsg_t nConstraints;
    unsg_t nEqConstraints;
    unsg_t newActiveIndex;
    unsg_t rptInterval;
    unsg_t singularIndex;
    unsg_t dualIteration;
    DualLoopExitStatus dualExitStatus;
    PrimalLoopExitStatus primalExitStatus;
    double gamma;
    double gammaCorrection;
    double styGamma;
    double scaleFactorDB;
    double rsNorm;
    double newActive;
    double dualTolerance;
    double dualityGap;
    double cost;
    CoreSettings settings;
    WorkSpace ws;
    std::unique_ptr<iTimer> timer;
    std::unique_ptr<Callback> uCallback;
    std::unique_ptr<ILinSolver> lSolver;
    std::unique_ptr<OrtScaler> ortScaler;
    SolverOutput output;
    InitStageStatus initStatus;
    std::vector<LinSolverTime> linSolverTimes;
    bool PrepareNNLS(const DenseQPProblem& problem);
    bool ComputeCholetsky(const matrix_t& M);
    bool OrigInfeasible();
    bool FullActiveSet();
    bool SkipCandidate(unsg_t indx);
    bool MakeLineSearch();
    bool IsCandidateForNewActive(unsg_t index, double toCompare, bool skip = true);
    void SetDefaultSettings();
    void TimeInterval(std::string& buf);
    void ScaleProblem();
    void ScaleD();
    void UnscaleD();
    void ComputeDualVariable();
    void UpdateGammaOnPrimalIteration();
    void UpdateGammaOnDualIteration();
    void AddToActiveSet(unsg_t indx);
    void RmvFromActiveSet(unsg_t indx);
    void ResetPrimal();
    void AllocateWs();
    void ExtendJacobian(const matrix_t& Jac, const std::vector<double>& b,
                        const std::vector<double>& lb, const std::vector<double>& ub);
    void ComputeOrigSolution();
    void ComputeExactLambdaOnActiveSet();
    void ComputeCost();
    void ComputeDualityGap();
    void ComputeViolationsExplicitly();
    void FillOutput();
    void SetInitData();
    void SetLinearSolver();
    void SetIterationData();
    void SetFinalData();
    void SetRptInterval();
    unsg_t SelectNewActiveComponent();
    unsg_t SolvePrimal();
    int UpdatePrimal();
};
}
#endif // CORE_H
