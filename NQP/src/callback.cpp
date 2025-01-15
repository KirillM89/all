#include "callback.h"
namespace QP_NNLS {
Callback1::Callback1(const std::string& filePath):
    filePath(filePath),
    logger(std::make_unique<Logger>()),
    logLevel(0u)
{ }

void Callback1::Init() {
    if (logLevel > 0u) {
        logger->SetFile(filePath);
    }
}
void Callback1::ProcessData(int stage) {
    // logLevel: 0 - write nothing, 1 - only final results,
    // 2 - only scalars, 3 - all the data
    if (logLevel == 0u) {
        return;
    }
    if (stage == 1) { // dump data after init stage
        logger->SetStage("INITIALIZATION");
        if(logLevel >= 2u) {
            logger->message("t Chol", initData.tChol);
            logger->message("t Inv", initData.tInv);
            logger->message("t M", initData.tM);
            logger->message("scale factor DB", initData.scaleDB);
        }
        if(logLevel >= 3u) {
            logger->dump("Choletsky", *initData.Chol);
            logger->dump("CholetskyInv", *initData.CholInv);
            logger->dump("Matrix M", *initData.M);
            logger->dump("vector s", *initData.s);
            logger->dump("vector c", *initData.c);
            logger->dump("vector b", *initData.b);
        }
    } else if (stage == 2) { // dump iteration data
        logger->message("---ITERATION---", iterData.iteration);
        if (logLevel >= 2u) {
            logger->message("new active component", iterData.newIndex,
                            "isSingular", iterData.singular ? 1 : 0, "gamma", iterData.gamma,
                            "dualTol", iterData.dualTol, "rsdNorm", iterData.rsNorm);
        }
        if (logLevel >= 3u) {
            logger->dump("active set", *iterData.activeSet);
            logger->dump("history", *iterData.activeSetHistory);
            logger->dump("zp", *iterData.zp);
            logger->dump("primal", *iterData.primal);
            logger->dump("dual", *iterData.dual);
        }
    }  else if (stage == 3) { // dump final data for any log level > 0
        logger->SetStage("RESULTS");
        if (finalData.dualStatus == DualLoopExitStatus::INFEASIBILITY) {
            logger->message("infeasibility");
        } else if (finalData.dualStatus == DualLoopExitStatus::ALL_DUAL_POSITIVE) {
            logger->message("all dual gt tolerance");
        } else if (finalData.dualStatus == DualLoopExitStatus::FULL_ACTIVE_SET) {
            logger->message("full active set");
        } else if (finalData.dualStatus == DualLoopExitStatus::ITERATIONS) {
            logger->message("iterations limit exceeded");
        } else {
            logger->message("convergence");
        }
        if (finalData.dualStatus != DualLoopExitStatus::INFEASIBILITY) {
           logger->dump("x", finalData.x);
           logger->message("cost", finalData.cost);
           logger->dump("lambda", *finalData.lambda);
           logger->dump("lambdaLw", *finalData.lambdaLw);
           logger->dump("lambdaUp", *finalData.lambdaUp);
           logger->dump("violations", *finalData.violations);
        }
    }
}
} // namespace QP_NNLS
