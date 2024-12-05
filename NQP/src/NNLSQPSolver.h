
#include <vector>
#include <queue>
#include <unordered_set>
#include <memory>
#include "types.h"
#include "log.h"
#include "timers.h"
#include "utils.h"
#include "decorators.h"
#ifndef NNLS_QP_SOLVER_H
#define NNLS_QP_SOLVER_H

namespace QP_NNLS {

struct NNLSQPResults {
	double residualNorm = 0.0;
	bool primalInfeasible = true;
};

class iDBScaler;

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
        std::vector<int> pmt;
        std::unordered_set<unsigned int> activeConstraints;
        std::unordered_set<unsigned int> negativeZp;
        matrix_t H;
        matrix_t M;
        matrix_t Jac;
        matrix_t Chol;
        matrix_t CholInv;
        std::queue<unsg_t> addHistory;
        void Clear();
    };

    enum PrimalRetStatus {
        SINGULARITY = 0x01,
        LINE_SEARCH_FAILED = 0x02
    };

public:
    Core() {}
    void Set(const CoreSettings& settings);
    void ResetProblem();
    void SetCallback(std::shared_ptr<Callback> callback);
    bool InitProblem(const DenseQPProblem& problem);
    void Solve();
    const SolverOutput& GetOutput() { return output; }
private:
    unsg_t nVariables;
    unsg_t nConstraints;
    unsg_t nEqConstraints;
    unsg_t newActiveIndex;
    DualLoopExitStatus dualExitStatus;
    PrimalLoopExitStatus primalExitStatus;
    double gamma;
    double styGamma;
    double scaleFactorDB;
    double rsNorm;
    double newActive;
    double dualTolerance;
    CoreSettings settings;
    WorkSpace ws;
    std::unique_ptr<iDBScaler> dbScaler;
    std::unique_ptr<iTimer> timer;
    std::shared_ptr<Callback> uCallback;
    SolverOutput output;
    bool PrepareNNLS(const DenseQPProblem& problem);
    bool OrigInfeasible();
    bool FullActiveSet();
    bool SkipCandidate(unsg_t indx);
    bool MakeLineSearch();
    void TimePoint(std::string& buf);
    void ScaleD();
    void ComputeDualVariable();
    void UpdateGammaOnDualIteration();
    void AddToActiveSet(unsg_t indx);
    void RmvFromActiveSet(unsg_t indx);
    void ResetPrimal();
    unsg_t SelectNewActiveComponent();
    unsg_t SolvePrimal();
    int UpdatePrimal();
};

class NNLSQPSolver
{
public:
	struct Settings {
		double residNormFeasibilityTol = 1.0e-12; //1.0e-12
		double origFeasibilityTol = 1.0e-6; // 1.0e-3
		double primalZero = -1.0e-7;
		double lsZero = 1.0e-14;
		int nPrimalIterations = 100;
		int nDualIterations = 130;  //100
		int logLevel = 3;
		double dualTol = -1.0e-14;
		DBScalerStrategy dbScalerStrategy = DBScalerStrategy::SCALE_FACTOR;
		CholPivotingStrategy cholPvtStrategy = CholPivotingStrategy::NO_PIVOTING;
	};

	NNLSQPSolver();
	virtual ~NNLSQPSolver() = default;
	bool Init(const ProblemSettings& settings);
	void Solve();
	const matrix_t& getChol() { return cholFactor;}
	const matrix_t& getCholInv() { return cholFactorInv;}
	const matrix_t& getH() { return H;}
	const std::vector<double>& getXOpt() { return xOpt;}
	double getCost() { return cost; }
	void Reset();
	const std::vector<double>& getLambda() { return lambdaOpt;}
	DualLoopExitStatus getDualStatus() { return dualExitStatus; }
	PrimalLoopExitStatus getPrimalStatus() { return primalExitStatus; }
	SolverExitStatus getExitStatus() { return solverExitStatus; }
	SolverOutput getOutput();
private:
	int nVariables = 0;
	int nConstraints = 0;
	int nLEqConstraints = 0;
	int nBounds = 0;
	int minDualIndex = -1;
	double gamma = 1.0;
	double styGamma = 0.0;
	double scaleFactor = 0.001;
	bool allActive = true;
	bool nextDualIteration = false;
	bool isBounds = false;
	double dualityGap = std::numeric_limits<double>::max();
	double maxConstrViolation = std::numeric_limits<double>::min();
	std::size_t activeSetSize = 0;
	DualLoopExitStatus dualExitStatus = DualLoopExitStatus::UNKNOWN;
	PrimalLoopExitStatus primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
	SolverExitStatus solverExitStatus = SolverExitStatus::UNKNOWN;
	Logger logger;
	const Settings st;
	matrix_t H;
	matrix_t A;
	matrix_t F;
	std::vector<int> activeSetIndices;
	std::vector<int> negativeZpIndices;
	std::vector<double> b;
	std::vector<double> d;
	std::vector<double> g;
	std::vector<double> primal;
	std::vector<double> dual;
	std::vector<double> zp;
	std::vector<double> vecS; // s = b + Mv
	std::vector<double> vecV; // v
	std::vector<double> lambdaOpt;
	std::vector<double> xOpt;
	std::vector<double> violations;
	std::vector<double> MtY;
	std::vector<int> pmtV; // vector of permutations computed in pivoting procedure i <-> pmtV[i]
	matrix_t matM; // M
	matrix_t cholFactor; // H = QT*Q
	matrix_t cholFactorInv; // Q^-1
	std::unique_ptr<iTimer> timer;
	SolverOutput output;
	CholetskyOutput choletskyOutput;
	std::unique_ptr<iDBScaler> dbScaler;
	double cost = std::numeric_limits<double>::max();
	double dualTolerance = 0;
	std::vector<int> singularConstraints;
	std::unordered_set<int> lastIndices;
	void setUserSettings(const UserSettings& settings);
	void initWorkSpace();
	void computeDualVariable(); 
	double computeResidualNorm(); 
	void updatePrimalAndActiveSet();
	bool solvePrimal();
	void makeLineSearch(const std::vector<int>& negativePrimalIndices);
	void getOrigSolution();
	bool origInfeasible();
	bool isOrigFeasible(double& violation);
	void computeDualityGap();
	bool prepareNNLS();
	void dumpProblem();
	void dumpNNLSDataStructures();
	void dumpUserSettings();
	void updateGammaOnDualIteration();
	void updateGammaOnPrimalIteration();
	double checkConstraints();
	void computeLambdaFromDualProblem();
	void extendJacWithBounds(const std::vector<double>& lw, const std::vector<double>& up);
	void findExactLambdaOnActiveSet();
	void scaleMB();
	void scaleD();
	void unscaleD();
	bool checkFinalSolution();
	double computeCost();
	bool wasAddedRecently(int iConstraint);
};

class LDL
{
public:
	// L*D*LT = A*AT
	LDL() = default;
	virtual ~LDL() = default;
	void Set(const matrix_t& A);
	void Compute();
	void Add(const std::vector<double>& row);
	void Remove(int i);
	const matrix_t& GetL();
	const std::vector<double>& GetD();
protected:
	int dimR = 0;
	int dimC = 0;
	int curIndex = 0;
	double d = 0.0;
	matrix_t L;
	std::vector<double> D;
	matrix_t A;
	std::vector<double> l;
	void compute_l();
	void compute_d();
	void update_L();
	void update_D();
	void solveLDb(const std::vector<double>& b, std::vector<double>& l);
	double getARowNormSquared(int row) const;
	void update_L_remove(int iRow, const matrix_t& Ltil);
	std::vector<int> activeRows;
};


class MMTbSolver
{
public:
	MMTbSolver() = default;
	virtual ~MMTbSolver() = default;
	int Solve(const matrix_t& M, const std::vector<double>& b);
	int nDZero();
	const std::vector<double>& GetSolution();
protected:
	void SolveForward(const matrix_t& L, const std::vector<double>& b);
	void SolveBackward(const std::vector<double>& D, const matrix_t& L);
	void GetMMTKernel(const std::vector<int>& dzeroIndices, const matrix_t& L,std::vector<double>& ker);
	//void ApplyPermutation(
	std::vector<double> solution;
	std::vector<double> forward;
	std::vector<double> backward;
	const double zeroTol = 1.0e-16;
	int ndzero = 0;
};

class iDBScaler {
public:
    virtual double Scale(const matrix_t& M, const std::vector<double>& s, const NNLSQPSolver::Settings& solverSettings ) = 0;
	virtual ~iDBScaler() = default;
	iDBScaler(const iDBScaler& other) = delete;
	iDBScaler& operator= (const iDBScaler& other) = delete;
    iDBScaler(iDBScaler&& other) = delete;
	iDBScaler& operator= (iDBScaler&& other) = delete;
protected:
    iDBScaler() = default;
};

class DBScaler: public iDBScaler {
public:
	DBScaler() = delete;
	DBScaler(DBScalerStrategy strategy); 
	virtual ~DBScaler() override = default;
	double Scale(const matrix_t& M, const std::vector<double>& s, const NNLSQPSolver::Settings& solverSettings) override;	
protected:
    const double maxBalanceFactor = 1.0; //1.0e3;
	const double balanceUpperBound = 1.0e30;
	const double balanceLowerBound = 1.0e-15;
	const double extremeFactorS = 1.0e-8;
	const double extremeFactorM = 1.0e8;
	DBScalerStrategy scaleStrategy = DBScalerStrategy::UNKNOWN;
};
}
#endif
