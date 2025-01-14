#include "decorators.h"
#include "core.h"
namespace QP_NNLS {

    QPNNLS::QPNNLS():
        core(std::make_unique<Core>()),
        isInitialized(false)
    { }
    QPNNLS::~QPNNLS() = default;

    void QPNNLS::Init(const Settings& settings) {
        if (VerifySettings(settings)) {
            core->Set(settings.coreSettings);
            isInitialized = true;
        }
    }
    void QPNNLS::SetCallback(std::unique_ptr<Callback> callback) {
        core->SetCallback(std::move(callback));
    }
    const SolverOutput& QPNNLS::GetOutput() {
        output = core->GetOutput();
        return output;
    }   
    bool QPNNLS::VerifySettings(const Settings& settings) {
        return true;
    }
    bool QPNNLSDense::SetProblem(const DenseQPProblem& problem) {
        if (!isInitialized) {
            return false;
        }
        core->ResetProblem();
        return core->InitProblem(problem);
    }
    void QPNNLSDense::Solve() {
        core->Solve();
    }
    InitStageStatus QPNNLSDense::GetInitStatus() {
        return core->GetInitStatus();
    }


}

