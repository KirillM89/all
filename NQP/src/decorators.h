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

    };

    class Observer {
    public:
        Observer() = default;
        virtual ~Observer() = default;
        virtual void processData() {
            return;
        }
    };

    class SolverCallback {
    public:
        SolverCallback() = default;
        virtual ~SolverCallback() = default;
        void setObserver(std::shared_ptr<Observer> observer) {
            this->observer = observer;
        }
    protected:
        std::shared_ptr<Observer> observer = nullptr;
    };

    class InitializationCallback {
    private:
        void getData(InitializationData& data);
    };

    class IterationCallback final: public SolverCallback {
    private:
        void getData(IterationData& data);
    };

    class FinalCallback final: public SolverCallback {
    private:
        void getData(FinalData& data);
    };

    class Core;
    class QPNNLS {
    public:
        void Init(const UserSettings& settings);
        void setObservers(std::shared_ptr<Observer> initObs,
                          std::shared_ptr<Observer> iterObs,
                          std::shared_ptr<Observer> finalObs);
        const SolverOutput& getOutput();
    protected:
        QPNNLS();
         ~QPNNLS() = default;
        QPNNLS(const QPNNLS& other);
        QPNNLS(QPNNLS&& other);
        OPNNLS& operator=(const QPNNLS& other);
        QPNNLS& operator=(QPNNLS&& other);

        std::unique_ptr<Core> core;
    };

    class QPNNLSDense : public QPNNLS {
    public:
        void Solve(const DenseQPProblem& problem);
    };

    class QPNNLSSparse : public QPNNLS {
    public:
        void Solve(const SparseQPProblem& problem);
    }

}

#endif // DECORATORS_H
