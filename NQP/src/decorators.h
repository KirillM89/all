#ifndef QP_NNLS_DECORATORS_H
#define QP_NNLS_DECORATORS_H
#include <memory>
#include <vector>
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
}

#endif // DECORATORS_H
