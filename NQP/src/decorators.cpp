#include "decorators.h"
#include "NNLSQPSolver.h"
namespace QP_NNLS {
    void InitializationCallback::getData(InitializationData& data) {

    }

    void IterationCallback::getData(IterationData& data){

    }

    void FinalCallback::getData(FinalData& data) {

    }

    QPNNLS::QPNNLS():
        core(std::make_unique<Core>()),
        isInitialized(false)
    { }

    void QPNNLS::Init(const Settings& settings) {
        if (VerifySettings(settings)) {
            core->Set(settings.coreSettings);
            isInitialized = true;
        }
    }

    void QPNNLS::setObservers(std::shared_ptr<Observer> initObs,
            std::shared_ptr<Observer> iterObs,
            std::shared_ptr<Observer> finalObs) {
            core->SetObservers(initObs, iterObs, finalObs);
    }

    const SolverOutput& QPNNLS::getOutput() {
        output = core->getOutput();
        return output;
    }

    void QPNNLSDense::Solve(const DenseQPProblem& problem) {
        core->ResetProblem();
        if (!isInitialized) {
            output.preprocStatus = PreprocStatus::INVALID_SETTINGS;
            return;
        }
        if (!core->InitProblem(problem)) {
            output.preprocStatus = PreprocStatus::INIT_FAILED;
            return;
        }
        core->Solve();
    }

    bool QPNNLS::VerifySettings(const Settings& settings) {
        return true;
    }



}

