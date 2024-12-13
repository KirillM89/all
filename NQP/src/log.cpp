#include "log.h"
namespace QP_NNLS {

	std::ostream& operator << (std::ostream& f, const matrix_t& mat) {
		#ifdef CPP_FORMAT 
		f << "{";
		#endif
		for (const auto& row : mat) {
			f << row;
			#ifdef CPP_FORMAT
	        if (&row != &mat.back()) {
				f << ",";
			}
			#endif
		}
		#ifdef CPP_FORMAT 
		f << "}\n";
		#endif

		return f;
	}

	bool Logger::SetFile(const std::string& fileName) {
		logFile = fileName;
		fid.open(logFile, std::ios::out);
		return fid.is_open();
	}

	void Logger::SetStage(const std::string& stageName) {
		const char* stageSep = "--------------";
		fid << "\n" << stageSep << stageName << stageSep << "\n";
	}
	void Logger::ToNewLine() {
		fid << "\n";
	}
	void Logger::PrintActiveSetIndices(const std::vector<int>& indices) {
		const int n = static_cast<int>(indices.size());
		fid << "active set indices: ";
		for (int indx = 0; indx < n; ++indx) {
			if (indices[indx] == 1) {
				fid << SEP << indx;
			}
		}
		fid << "\n";
	}

    Callback1::Callback1(const std::string& filePath):
        logger(std::make_unique<Logger>())
    {
        logger->SetFile(filePath);
    }


    void Callback1::ProcessData(int stage) {
        if (stage == 1) { // dump data after init stage
            logger->SetStage("INITIALIZATION");
            logger->dump("Choletsky", initData.Chol);
            logger->dump("CholetskyInv", initData.CholInv);
            logger->dump("Matrix M", initData.M);
            logger->dump("vector s", initData.s);
            logger->dump("vector c", initData.c);
            logger->dump("vector b", initData.b);
            logger->message("t Chol", initData.tChol);
            logger->message("t Inv", initData.tInv);
            logger->message("t M", initData.tM);
            logger->message("scale factor DB", initData.scaleDB);
        } else if (stage == 2) { // dump iteration data
            logger->message("---ITERATION---", iterData.iteration);
            logger->dump("active set", *iterData.activeSet);
            logger->dump("history", *iterData.activeSetHistory);
            logger->dump("zp", *iterData.zp);
            logger->dump("primal", *iterData.primal);
            logger->dump("dual", *iterData.dual);
            logger->message("new active component", iterData.newIndex,
                            "isSingular", iterData.singular ? 1 : 0, "gamma", iterData.gamma,
                            "dualTol", iterData.dualTol, "rsdNorm", iterData.rsNorm);
        }  else if (stage == 3) { // dump final data
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
               logger->dump("lambda", finalData.lambda);
               logger->dump("lambdaLw", finalData.lambdaLw);
               logger->dump("lambdaUp", finalData.lambdaUp);
               logger->dump("violations", finalData.violations);
            }
        }
    }


}
