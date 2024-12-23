
#include <vector>
#include <list>
#include <string>
#include "timers.h"
#ifndef NNLS_QP_SOLVER_TYPES_H
#define NNLS_QP_SOLVER_TYPES_H
namespace QP_NNLS {
using matrix_t = std::vector<std::vector<double>>;
using unsg_t = unsigned int;
namespace CONSTANTS {
	constexpr double cholFactorZero = 1.0e-14;
	constexpr double pivotZero = 1.0e-14;
}
static_assert(CONSTANTS::cholFactorZero > 0.0);
static_assert(CONSTANTS::pivotZero > 0.0);

struct CholetskyOutput {
    std::list<std::pair<int, double>> negativeDiag; // negative diagonal elements in range [-cholFactorZero, 0)
	double negativeBlocking = 1.0; // neagtive diagonal element which block all the next steps
	bool pivoting = false;
};

constexpr double gammaBegin = 1.0;

enum class SolverRetStatus {
	SUCCESS = 0,
	FAIL,
};
enum class ProblemConfiguration {
	DENSE = 0,
	SPARSE,
};
enum class GammaUpdateStrategyPrimal {
	NO_UPDATE = 0,
	DECREMENT_BY_D_NORM
};

enum class DBScalerStrategy {
	SCALE_FACTOR, 
	BALANCE,
	UNKNOWN
};

enum class LinSolverType {
    CUMULATIVE_LDLT = 0,
    CUMULATIVE_EG_LDLT,
    DYNAMIC_LDLT,
};

enum class CholPivotingStrategy {
	NO_PIVOTING,
	FULL,
	PARTIAL,
	UNKNOWN
};

enum class GammaUpdateStrategyDual {
	NO_UPDATE = 0,
	INCREMENT_BY_S_COMPONENT,
};

enum class InitStageStatus {
    SUCCESS = 0,
    CHOLETSKY
};

struct LinSolverOutput {
    bool emptyInput = false;
    unsg_t nDNegative = std::numeric_limits<unsg_t>::max(); // number of d<=0 in LDLT
    std::vector<double> solution;
    std::list<unsg_t> indices;
};

struct ActiveSetUpdateSettings {
    int rptInterval = 0;
    bool rejectSingular = false;
    bool firstInactive = true;
};

struct CoreSettings {
    LinSolverType linSolverType = LinSolverType::CUMULATIVE_LDLT;
    //LinSolverType linSolverType = LinSolverType::CUMULATIVE_EG_LDLT;
    DBScalerStrategy dbScalerStrategy = DBScalerStrategy::SCALE_FACTOR;
    CholPivotingStrategy cholPvtStrategy = CholPivotingStrategy::NO_PIVOTING;
    unsg_t nDualIterations = 1000;
    unsg_t nPrimalIterations = 100;
    double nnlsResidNormFsb = 1.0e-16;
    double origPrimalFsb = 1.0e-6;
    double nnlsPrimalZero = -1.0e-7; //-1.0e-7; //zp < 0 => zp < nnlsPrimalZero
    double minNNLSDualTol = -1.0e-12;
    double prLtZero = 1.0e-14;
    bool gammaUpdate = true;
    ActiveSetUpdateSettings actSetUpdtSettings;
};

struct Settings {
    bool checkCoreSettings = false;
    CoreSettings coreSettings;
};


struct UserSettings {
	ProblemConfiguration configuration = ProblemConfiguration::DENSE;
	DBScalerStrategy dbScalerStrategy = DBScalerStrategy::SCALE_FACTOR;
	CholPivotingStrategy cholPvtStrategy = CholPivotingStrategy::NO_PIVOTING;
	int nDualIterations = 100;
	int nPrimalIterations = 100;
    int logLevel = 3;
	double nnlsResidNormFsb = 1.0e-16;
	double origPrimalFsb = 1.0e-6;
	double nnlsPrimalZero = -1.0e-16; // -1.0e-12; //zp < 0 => zp < nnlsPrimalZero
	double minNNLSDualTol = -1.0e-12;
	std::string logFile = "logNNLS.txt";
	bool checkProblem = false;
};

struct DenseQPProblem {
	//1/2xtHx + cx
	//Ax <= b
	//Fx = g
	matrix_t H;
    matrix_t A;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> up;
    std::vector<double> lw;
    unsg_t nEqConstraints;
};

struct SparseQPProblem {

};

struct ProblemSettings {
	ProblemSettings() = delete;
	ProblemSettings(const UserSettings& settings, const DenseQPProblem& problem):
		uSettings(settings), problemD(problem)
	{};
	virtual ~ProblemSettings() = default;
	UserSettings uSettings;
	const DenseQPProblem& problemD;
};


enum class DualLoopExitStatus {
    ALL_DUAL_POSITIVE = 0,
	FULL_ACTIVE_SET,
	ITERATIONS,
	INFEASIBILITY,
	UNKNOWN
};

enum class PrimalLoopExitStatus {
    EMPTY_ACTIVE_SET = 0,
	ALL_PRIMAL_POSITIVE,
	ITERATIONS,
	EMPTY_ACTIVE_SET_ON_ZERO_ITERATION,
	SINGULAR_MATRIX,
	DIDNT_STARTED,
    LINE_SEARCH_FAILED,
	UNKNOWN
};

enum class SolverExitStatus {
	CONSTRAINTS_VIOLATION = 0,
	SOLVE_PRIMAL,
	SUCCESS,
	UNKNOWN
};

enum class PreprocStatus {
    SUCCESS = 0,
    INVALID_SETTINGS,
    INIT_FAILED
};

struct SolverOutput {
    DualLoopExitStatus dualExitStatus;
    PrimalLoopExitStatus primalExitStatus;
    unsg_t nDualIterations;
	double maxViolation;
	double dualityGap;
    double cost;
    std::vector<double> x;
    std::vector<double> lambda;
    std::vector<double> lambdaLw;
    std::vector<double> lambdaUp;
    std::vector<double> violations;
};

}
#endif
