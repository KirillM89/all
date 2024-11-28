#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "NNLSQPSolver.h"
#include "utils.h"
namespace QP_NNLS {
	NNLSQPSolver::NNLSQPSolver():
		nVariables(0),
		nConstraints(0),
		nLEqConstraints(0),
		nBounds(0),
		minDualIndex(-1),
		gamma(gammaBegin),
		styGamma(0.0),
		allActive(true),
		nextDualIteration(false),
		isBounds(false),
		dualityGap(std::numeric_limits<double>::max()),
		maxConstrViolation(std::numeric_limits<double>::min()),
		activeSetSize(0),
		dualExitStatus(DualLoopExitStatus::UNKNOWN),
		primalExitStatus(PrimalLoopExitStatus::UNKNOWN),
		H{}, A{}, F{},
		activeSetIndices{},
		b{}, d{}, g{},
		primal{}, dual{},
		zp{}, vecS{}, vecV{},
		lambdaOpt{},
		xOpt{},
		violations{},
		MtY{},
		pmtV{},
		matM{},
		cholFactor{},
		cholFactorInv{},
		dbScaler(nullptr),
		cost(std::numeric_limits<double>::max())
	{};

	bool NNLSQPSolver::Init(const ProblemSettings& settings) {
		Reset();
		//xtHx + dx; Ax <= b
		setUserSettings(settings.uSettings);
		timer = std::make_unique<wcTimer>();
		timer->Start();
		dbScaler = std::make_unique<DBScaler>(st.dbScalerStrategy);
		if (!logger.SetFile(settings.uSettings.logFile)) {
			return false;
		}
		if (settings.uSettings.checkProblem) {
			//TODO
		} else {
			H = settings.problemD.H;
			d = settings.problemD.c;
			A = settings.problemD.A;
			b = settings.problemD.b;
			F = settings.problemD.F;
			g = settings.problemD.g;
			nVariables = static_cast<int>(H.size());
			nConstraints = static_cast<int>(A.size());
			nLEqConstraints = static_cast<int>(F.size());
			extendJacWithBounds(settings.problemD.lw, settings.problemD.up);
			initWorkSpace();
			if(!prepareNNLS()) {
				return false;
			}
			scaleFactor = dbScaler -> Scale(matM, vecS, st);
			logger.message("scaleFactor", scaleFactor);
			logger.message("unscaled problem:");
			dumpProblem();
			scaleD();
			if (settings.uSettings.logLevel > 0) {
				dumpProblem();
				dumpNNLSDataStructures();
			}
	    }
		TimeIntervals tIntervals;
		timer->toIntervals(timer->TimeFromStart(), tIntervals);
		logger.message("Init stage time ", tIntervals.minutes, "sec", tIntervals.sec, "ms", tIntervals.ms, "mus", tIntervals.mus);
		return true;

	}

	void NNLSQPSolver::extendJacWithBounds(const std::vector<double>& lw, const std::vector<double>& up) {
		if (up.size() != lw.size() || up.size() != nVariables) {
			return;
		}
		isBounds = true;
		b.resize(b.size() + 2 * nVariables, 0.0);
		A.resize(A.size() + 2 * nVariables, std::vector<double>(nVariables, 0.0));
		for (int i = 0; i <  nVariables; ++i) {
			const int ibg = 2 * i + nConstraints;
			A[ibg][i] = 1.0;
			A[ibg + 1][i] = -1.0;
			b[ibg] = up[i];
			b[ibg + 1] = -lw[i];
		}
		nConstraints += 2 * nVariables;
	}

	void NNLSQPSolver::setUserSettings(const UserSettings& settings) {
		const_cast<int&>(st.nDualIterations) = settings.nDualIterations;
		const_cast<int&>(st.nPrimalIterations) = settings.nPrimalIterations;
		const_cast<double&>(st.origFeasibilityTol) = settings.origPrimalFsb;
		const_cast<double&>(st.residNormFeasibilityTol) = settings.nnlsResidNormFsb;
		const_cast<int&>(st.logLevel) = settings.logLevel;
		const_cast<double&>(st.primalZero) = settings.nnlsPrimalZero;
		const_cast<DBScalerStrategy&>(st.dbScalerStrategy) = settings.dbScalerStrategy;
		const_cast<CholPivotingStrategy&>(st.cholPvtStrategy) = settings.cholPvtStrategy;
	}

	bool NNLSQPSolver::prepareNNLS() {
		std::vector<double> MbyV(nConstraints);
		TimeIntervals tIntervals;
		ticks_t t = timer->TimeFromStart();
		timer->toIntervals(t, tIntervals);
		if (st.cholPvtStrategy == CholPivotingStrategy::NO_PIVOTING) {
			if (!ComputeCholFactorT(H, cholFactor, choletskyOutput)) {  // H = L_T * L  
				return false;
			}
		} else if (st.cholPvtStrategy == CholPivotingStrategy::FULL) {
			// full pivoting:
            // x == P * x_n;  P - permuation matrix
			// 0.5 * x_T * H * x + c * x = 0.5 * x_n_T * P_T * H * P * x_n + c_T * P * x_n = 0.5 * x_n_T * H_n * x_n + c_n_T * x_n
			// H_n = P_T * H * P ; c_n = P_T * c
			// A * x <= b  A * P *x_n <= b  A_n = A * P   A_n * x_n <= b  
			int ret = ComputeCholFactorTFullPivoting(H, cholFactor, pmtV); // H -> H_n
			if (ret != 0) {
				logger.message("Hessian is singular rk=", ret, " sz=", nVariables);
				return false;
			}
			logger.dump("permutation vector", pmtV);
			PermuteColumns(A, pmtV);
			PTV(d, pmtV);
		}
		timer->toIntervals(timer->TimeFromStart() - t , tIntervals);
		logger.message("Chol factor time sec", tIntervals.sec);
		t = timer->TimeFromStart();
		InvertTriangle(cholFactor, cholFactorInv);   // Q^-1
		timer->toIntervals(timer->TimeFromStart() - t, tIntervals);
		logger.message("InvTriangle factor time sec", tIntervals.sec);
		t = timer->TimeFromStart();
		Mult(A, cholFactorInv, matM);                // M = A*Q^-1   nConstraints x nVariables
		timer->toIntervals(timer->TimeFromStart() - t, tIntervals);
		logger.message("Matrix M time sec", tIntervals.sec);
		t = timer->TimeFromStart();
		MultTransp(cholFactorInv, d, vecV);          // v = Q^-T * d nVariables
		timer->toIntervals(timer->TimeFromStart() - t, tIntervals);
		logger.message("Vector V time sec", tIntervals.sec);
		t = timer->TimeFromStart();
		Mult(matM, vecV, MbyV);                      // M*v      nConstraints  
		timer->toIntervals(timer->TimeFromStart() - t, tIntervals);
		logger.message("M * v time sec", tIntervals.sec);
		VSum(MbyV, b, vecS);						 // s=b+Mv   nConstraints
		return true;
	}

	void NNLSQPSolver::dumpProblem() {
		logger.SetStage("PROBLEM");
		logger.message("1/2 xT * H * x + d * x, s.t. A * x <= b");
		logger.message("number of variables:", H.size());
		logger.message("number of constraints:", A.size());
		logger.dump("matrix H", H);
		logger.dump("vector d", d);
		logger.dump("matrix A", A);
		logger.dump("vector b", b);
	}
	void NNLSQPSolver::dumpNNLSDataStructures() {
		logger.SetStage("INITIALIZATION");
		logger.dump("choletsky factor Q", cholFactor);
		logger.dump("choletsky factor invert Q-1", cholFactorInv);
		logger.dump("matrix M = A * Q-1", matM);
		logger.dump("vector v = Q-T * d", vecV);
		logger.dump("vector s = b + M * v", vecS);
	}
	void NNLSQPSolver::initWorkSpace() {
		primal = std::vector<double>(nConstraints);
		dual = std::vector<double>(nConstraints);
		cholFactor = matrix_t(nVariables, std::vector<double>(nVariables));
		cholFactorInv = matrix_t(nVariables, std::vector<double>(nVariables, 0.0));
		matM = matrix_t(nConstraints, std::vector<double>(nVariables, 0.0));
		vecV = std::vector<double>(nVariables);
		activeSetIndices = std::vector<int>(nConstraints, 0);
		violations = std::vector<double>(nConstraints, 0);
		vecS = std::vector<double>(nConstraints);
		lambdaOpt = std::vector<double>(nConstraints);
		xOpt = std::vector<double>(nVariables);
		zp = std::vector<double>(nConstraints);
		negativeZpIndices = std::vector<int>(nConstraints, -1);
		MtY = std::vector<double>(nVariables);
		if (st.cholPvtStrategy == CholPivotingStrategy::FULL) {
			pmtV.resize(nVariables, -1);
		}
	}

	void NNLSQPSolver::computeDualVariable() {
		double styGm = gamma;
		MultTransp(matM, primal, activeSetIndices, MtY); // MTy
		Mult(matM, MtY, dual); // MMTy
		styGm += DotProduct(vecS, primal, activeSetIndices); //gamma + sTy
		for (int i = 0; i < nConstraints; ++i) {
			dual[i] += styGm * vecS[i];   // MMTy + (gamma + sTy)s
		}
		for (int i = 0; i < nConstraints; ++i) {
			if (activeSetIndices[i] == 1) {
				dual[i] = 0.0;
			}
		}
		styGamma = gamma + DotProduct(vecS, primal);
	}

	void NNLSQPSolver::scaleMB() {
		for (int i = 0; i < nConstraints - 2 * nVariables; ++i) {
			//double norm = sqrt(DotProduct(matM[i], matM[i]));;
			for (double& el : matM[i]) {
				el /= b[i];
			}
			b[i] = 1.0;
		}
	}
	void NNLSQPSolver::scaleD() {
		for (int i = 0; i < nConstraints; ++i) {
			b[i] *= scaleFactor;
			vecS[i] *= scaleFactor;
		}
		for (int i = 0; i < nVariables; ++i) {
			d[i] *= scaleFactor;
			vecV[i] *= scaleFactor;
		}
	}

	void NNLSQPSolver::unscaleD() {
		for (int i = 0; i < nConstraints; ++i) {
			b[i] /= scaleFactor;
			lambdaOpt[i] /= scaleFactor;
		}
		for (int i = 0; i < nVariables; ++i) {
			xOpt[i] /= scaleFactor;
			d[i] /= scaleFactor;
			if (st.dbScalerStrategy == DBScalerStrategy::SCALE_FACTOR && scaleFactor < 1.0) {
				const_cast<double&>(st.origFeasibilityTol) /= scaleFactor;
			}
		}
	}

	void NNLSQPSolver::updateGammaOnDualIteration() {

	}

	void NNLSQPSolver::updateGammaOnPrimalIteration() {

	}

	bool NNLSQPSolver::origInfeasible() {
		double norm = computeResidualNorm();
		if (st.logLevel > 1) {
			logger.message("residual norm", norm);
		}
		return (norm < st.residNormFeasibilityTol);
	}

	double NNLSQPSolver::computeResidualNorm() {
		MultTransp(matM, primal, activeSetIndices, MtY); // MTy
		styGamma = gamma + DotProduct(vecS, primal, activeSetIndices);
		return DotProduct(MtY, MtY) + styGamma * styGamma; //yTMMTy + (gamma + sTy)
	}

	bool NNLSQPSolver::solvePrimal() {
		// output zp
		// solve (M * MT + s * sT) * zp = -gamma * s on active set 
		// zp not on active set == 0
		matrix_t M;
		std::vector<double> s;
		// [M s]*[M_T / s_T] = -gamma * s 
		if (activeSetSize > 0) {
			for (int i = 0; i < nConstraints; ++i) {
				if (activeSetIndices[i] == 1) {
					M.push_back(matM[i]);
					M.back().push_back(vecS[i]);
					s.push_back(-gamma * vecS[i]);
				}
			}
		} else {
			for (int i = 0; i < nConstraints; ++i) {
					M.push_back({vecS[i]});
					s.push_back(-gamma * vecS[i]);
				}
		}
		MMTbSolver mmtb;
		if (mmtb.Solve(M, s) > 0) {
			//return false;
		}
		std::vector<double> solActiveSet = mmtb.GetSolution();
		int iactive = 0;
		for (int i = 0; i < nConstraints; ++i) {
			if (activeSetIndices[i] == 1) {
				zp[i] = solActiveSet[iactive++];
			} else {
				zp[i] = 0.0;
			}
		}
		if (st.logLevel >= 3) {
			logger.message("primal solution quality:");
			matrix_t MMT(M.size(), std::vector<double>(M.size(), 0.0));
			M1M2T(M, M, MMT);
			std::vector<double> MMTz(M.size());
			Mult(MMT, solActiveSet, MMTz);
			for (int i = 0; i < M.size(); ++i) {
				logger.message("i", i,  MMTz[i] - s[i]);
			}
		}
		return true;
	}

	void NNLSQPSolver::makeLineSearch(const std::vector<int>& negativePrimalIndices) {
		double minStep = std::numeric_limits<double>::max();
		const double initStep = minStep;
		// case if all zp are non-negative must be proccessed before this function, negativePrimalIndices must not be empty
		for (int indx : negativePrimalIndices) {
			if (indx == -1) {
				continue;
			}
			double denominator = primal[indx] - zp[indx]; 
			if (!isSame(denominator, 0.0)) {
				minStep = std::fmin(minStep, primal[indx] / denominator);
			}
		}
		if (isSame(minStep, initStep)) {
			// final step == initial step if current primal and zp are same for negative zp components 
			// primal must always be >= 0, so this case can be if zp[i] == 0 but in this function all zp components are negative 
			if (st.logLevel > 1) {
				logger.message("Line search WARNING: step didn't found");
			}
			return; // ???
		}
		if (st.logLevel > 1) {
			logger.message("Line search min step", minStep);
		}
		//y_next = y + step * (z - y)
		double gammaCorrection = 0.0;
		for (int i = 0; i < nConstraints; ++i) {
			primal[i] += minStep * (zp[i] - primal[i]);
			if (std::fabs(primal[i]) < st.lsZero) {
				gammaCorrection += std::fabs(vecS[i]);
				if (activeSetIndices[i] != 0) {
					activeSetSize--;
					activeSetIndices[i] = 0;
				}
			}
		}
		updateGammaOnPrimalIteration();
		if (dualTolerance < 1.0e-10) {
			gamma = std::fabs(gamma - gammaCorrection);
		}
	}

	double NNLSQPSolver::checkConstraints() {
		Mult(A, xOpt, violations);
		double maxViolation = std::numeric_limits<double>::min();
		int iMaxViolated = -1;
		for (int ic = 0; ic < nConstraints; ++ic) {
			double vl = violations[ic] - b[ic];
			if (st.logLevel > 1) {
				if (vl > st.origFeasibilityTol) {
					logger.message("violation: ", ic, vl);
					if (vl > maxViolation) {
						maxViolation = vl;
						iMaxViolated = ic;
					}
				} else {
					logger.message("constraint ", ic, "Ax_i - b_i=", vl);
				}
			}
		}
		if (st.logLevel > 1) {
			logger.dump("x optimal", xOpt);
			logger.dump("cost", computeCost());
		}
		return maxViolation;
	}

	void NNLSQPSolver::getOrigSolution() {
        
		double sty = DotProduct(vecS, primal, activeSetIndices);
		double lambdaTerm = -1.0 / (gamma + sty);
		if (st.logLevel > 1) {
		    double gap = DotProduct(primal, dual) * lambdaTerm * lambdaTerm;
			logger.message("lambda term = ", lambdaTerm, " gap=", gap);
		}
		for (int i = 0; i < nConstraints; ++i) {
			lambdaOpt[i] = lambdaTerm * primal[i] ;
		}
		findExactLambdaOnActiveSet();
		std::vector<double> u(nVariables, 0.0);
		if (activeSetSize > 0) {
			MultTransp(matM, lambdaOpt, activeSetIndices, u);
		}
		std::vector<double> u_v(nVariables);
		for (int i = 0; i < nVariables; ++i) {
			u_v[i] = u[i] - vecV[i];
		}
		Mult(cholFactorInv, u_v, xOpt);
		for (int i = 0; i < nConstraints; ++i) {
			lambdaOpt[i] = -lambdaOpt[i];
		}
		computeDualityGap();
	}

	void NNLSQPSolver::computeLambdaFromDualProblem() {
		Mult(A, xOpt, lambdaOpt);
		for (int i = 0; i < nConstraints; ++i) {
			lambdaOpt[i] = -vecS[i] - lambdaOpt[i] + b[i];
		}
		std::vector<double> r = lambdaOpt;
		matrix_t MMT(nConstraints, std::vector<double>(nConstraints, 0.0));
		for (int i = 0; i < nConstraints; ++i) {
			for (int j = 0; j < nConstraints; ++j) {
				for (int k = 0; k < nVariables; ++k) {
					MMT[i][j] += matM[i][k] * matM[j][k];
				}
			}
		}
		MMTbSolver solver;
		solver.Solve(MMT, r);
		lambdaOpt = solver.GetSolution();
	}

	bool NNLSQPSolver::isOrigFeasible(double& violation) {
		getOrigSolution();
		std::vector<double> Ax(A.size());
		Mult(A, xOpt, Ax);
		for (int i = 0; i < nConstraints; ++i) {
			violation = Ax[i] - b[i];
			if (violation > st.origFeasibilityTol) {
				return false;
			}
		}
		return true;
	}
	double NNLSQPSolver::computeCost() {
		double cost = DotProduct(d, xOpt);
		for (int i = 0; i < nVariables; ++i) {
			for (int j = 0; j < i; ++j) {
				cost += H[i][j] * xOpt[i] * xOpt[j];
			}
			cost += 0.5 * H[i][i] * xOpt[i] * xOpt[i];
		}
		return cost;
	}
	void NNLSQPSolver::findExactLambdaOnActiveSet() {
		// Correct lambdas for active constraints to improve feasibility
		matrix_t M;
		std::vector<double> s;
		for (int i = 0; i < nConstraints; ++i) {
			//if (lambdaOpt[i] != 0.0) { 
			if (activeSetIndices[i] == 1) {
				M.push_back(matM[i]);
				s.push_back(vecS[i]);
			}
		}
		if (M.empty()) {
			return;
		}
		MMTbSolver mmtb;
		mmtb.Solve(M, s);
		std::vector<double> solActiveSet = mmtb.GetSolution();
		int ii = 0;
		for (int i = 0; i < nConstraints; ++i) {
			if (activeSetIndices[i] == 1) {
				lambdaOpt[i] = solActiveSet[ii++];
			}
		}

	}
	void NNLSQPSolver::computeDualityGap() {
		// For original problem
		// A * x_opt - b = -s - M * M_T * lambda
		// Compute -s - M * M_T * lambda
		std::vector<double> MMTLm(nConstraints);
		std::vector<double> violations(nConstraints);
		MultTransp(matM, lambdaOpt, MtY);
		Mult(matM, MtY, MMTLm);
		VSum(MMTLm, vecS, violations);
		const double lamTByS = DotProduct(lambdaOpt, vecS);
		const double vTv = DotProduct(vecV, vecV);
		const double mty2 = DotProduct(MtY, MtY);
		const double dualValue = -0.5 * (mty2 + vTv) - lamTByS;
		std::vector<double> Ax(nConstraints);
		Mult(A, xOpt, Ax);
		const double fsb = DotProduct(Ax, lambdaOpt) - DotProduct(b, lambdaOpt);
		const double primalValue = computeCost() + fsb;
		dualityGap = primalValue - dualValue;
		if (st.logLevel > 2) {
			logger.message("pr", primalValue, "dl", dualValue, "gap", dualityGap);
		}
	}
	void NNLSQPSolver::Solve() {
		if (st.logLevel > 1) {
			logger.SetStage("SOLVE");
			logger.message("solving NNLS QP problem. Primal and dual variables are for QPNNLS not for original one");
			logger.PrintActiveSetIndices(activeSetIndices);
			logger.dump("initial active set", activeSetIndices);
		}

		minDualIndex = -1;            // index of most negative NNLS dual variable
		int singularConstraint = -1;  // index of constraint which makes [M s] matrix singular
		int dualIteration = 0;
		//dual loop for NNLS problem
		dualExitStatus = DualLoopExitStatus::UNKNOWN;
		primalExitStatus = PrimalLoopExitStatus::DIDNT_STARTED;
		int iLast = -1;
		while (dualIteration < st.nDualIterations) {
			dualExitStatus = DualLoopExitStatus::UNKNOWN;
			if (st.logLevel > 1) {
				logger.message("--------dual iteration:", dualIteration, "------------");
			}
			/*if (singularConstraint != -1) {
				activeSetIndices[singularConstraint] = 0; 
				--activeSetSize;
				singularConstraint = -1;
			}*/
			// first chek feasiblity of original problem
			// if active set is empty residual norm = gamma ^ 2 > 0
			// if active set size > 0 then infeasibility take place when residual norm = 0
			if (activeSetSize > 0) {   
				if (origInfeasible()) {
					if (st.logLevel > 1) {
						logger.message("Infeasibility detected");
					}
					dualExitStatus = DualLoopExitStatus::INFEASIBILITY;
					break;
				}
			}
			//check if active set is full
			allActive = true;
			for (const auto& indx : activeSetIndices) {
				if (indx == 0) {
					allActive = false;
					break;
				}
			}
			if (allActive) {
				if (st.logLevel > 1) {
					logger.message("STOP: active set is full");
				}
				dualExitStatus = DualLoopExitStatus::FULL_ACTIVE_SET;
				break;
			}
			//compute dual variable on active set
			computeDualVariable();
			const double minTol= st.dualTol; // <= 0
			dualTolerance = -styGamma * st.origFeasibilityTol;
			dualTolerance = (dualTolerance >= 0.0) ? fmax(dualTolerance, -minTol) : fmin(dualTolerance, minTol);
			if (st.logLevel > 2) {
				logger.dump("dual variable:", dual);
				logger.message("dual tolerance:", dualTolerance);
			}
			double minDual = std::numeric_limits<double>::max();
			minDualIndex = -1;
            //first: iterate through all inactive constraints to find most negative dual variable  
			for (int i = 0; i < nConstraints; ++i) {
				if (activeSetIndices[i] == 1 || 
				    std::find(singularConstraints.begin(), singularConstraints.end(), i) != singularConstraints.end() ||
					wasAddedRecently(i)) {  // also skip constraint which leads to [M s] singularity. Singular constraint must be removed from active set
					continue;
				}
				const double dl = dual[i];
				if (dl < dualTolerance && dl < minDual) {
					minDual = dl;
					minDualIndex = i;
				}
			}

			// second: if active set has not been updated with inactive constraints, try to update it with already active
			// cycling may take place in this case 
			if (minDualIndex == -1) {
				for (int i = 0; i < nConstraints; ++i) {
					if (activeSetIndices[i] == 1 && i != iLast ) {
						const double dl = dual[i];
						if (dl < dualTolerance && dl < minDual) {
							minDual = dl;
							minDualIndex = i;
						}
					}
				}
			}		
			// 
			if (minDualIndex >= 0) {
				lastIndices.insert(minDualIndex);
				logger.message("new index", minDualIndex);
				if (activeSetIndices[minDualIndex] == 0) {
					activeSetIndices[minDualIndex] = 1;
					++activeSetSize;
				}
				updateGammaOnDualIteration();
				if (dualTolerance < 1.0e-10) {
					gamma += std::fabs(vecS[minDualIndex]);
				}
			}
			else {
				if (st.logLevel > 1) {
					logger.message("STOP: all dual > tol");
				}
				//std::fill(activeSetIndices.begin(), activeSetIndices.end(), 1);
				//activeSetSize = nConstraints;
				dualExitStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
				break;
			}
			if (st.logLevel > 2) {
				logger.dump("active set after dual:", activeSetIndices);
			}
			if (st.logLevel > 1) {
				logger.message("gamma", gamma);
			}
			//updated active set, computed dual variables for new active set, updated gamma => solve primal problem 
			int primalIteration = 0;
			while (primalIteration < st.nPrimalIterations) {
				primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
				if (st.logLevel > 1) {
					logger.message("primal iteration", primalIteration);
				}
				// active set may be empty only when primal loop made 1 or more iterations
				bool activeSetEmpty = true;
				for (int indx : activeSetIndices) {
					if (indx == 1) {
						activeSetEmpty = false;
						break;
					}
				}
				if (activeSetEmpty) {
					primalExitStatus = primalIteration == 0 ? PrimalLoopExitStatus::EMPTY_ACTIVE_SET_ON_ZERO_ITERATION : PrimalLoopExitStatus::EMPTY_ACTIVE_SET;
					break;
				}
				if (st.logLevel > 2) {
					logger.dump("primal before update:", primal);
				}
				ticks_t tStart = timer->TimeFromStart();
				// active set must be non-empty before solvePrimal 
				if (!solvePrimal()) { 
					singularConstraints.push_back(minDualIndex);
					primalExitStatus = PrimalLoopExitStatus::SINGULAR_MATRIX;
					activeSetIndices[minDualIndex] = 0;
					--activeSetSize;
					break;
				} else {
					singularConstraints.clear();
				}
				TimeIntervals tIntervals;
				timer->toIntervals(timer->TimeFromStart() - tStart, tIntervals);
				if (st.logLevel > 1) {
					logger.message("solve primal time: min", tIntervals.minutes, "sec", tIntervals.sec, "ms", tIntervals.ms, "mus", tIntervals.mus);
				}

				if (st.logLevel > 2) {
					logger.dump("primal on active set:", zp);
				}
			
				bool allNonNegative = true;
				std::fill(negativeZpIndices.begin(), negativeZpIndices.end(), -1);
				for (int i = 0; i < nConstraints; ++i) {
					if (zp[i] < st.primalZero) { // st.primalZero must be <= 0 
						allNonNegative = false;
						negativeZpIndices[i] = i;
					}
				}
				//all zp >= 0 means that optimal solution found for original problem, but go to next dual iteration to improve possible infeasibilities 
				if (allNonNegative) {
					primal = zp;
					primalExitStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
					break;
				}
				makeLineSearch(negativeZpIndices);
				if (st.logLevel > 2) {
					logger.dump("primal after update:", primal);
					logger.dump("active set after primal update:", activeSetIndices);
				}
				if (st.logLevel > 1) {
					logger.message("gamma", gamma);
				}
				++primalIteration;
			}
			if (primalExitStatus == PrimalLoopExitStatus::EMPTY_ACTIVE_SET_ON_ZERO_ITERATION) {
				return; // unexpected behaviour 
			}
			if (primalIteration >= st.nPrimalIterations) {
				primalExitStatus = PrimalLoopExitStatus::ITERATIONS;
			}
			++dualIteration;
		    if (!singularConstraints.empty()) {
				logger.message("singular constraint");
				continue;
			}
			if (st.logLevel > 2) {
				getOrigSolution();
				checkConstraints();
			}
		}
		if (dualIteration >= st.nDualIterations) {
			dualExitStatus = DualLoopExitStatus::ITERATIONS;
		}
		if (dualExitStatus == DualLoopExitStatus::UNKNOWN || primalExitStatus == PrimalLoopExitStatus::UNKNOWN 
			|| primalExitStatus == PrimalLoopExitStatus::EMPTY_ACTIVE_SET_ON_ZERO_ITERATION) {
			// this case must't take place, unexpected behavior
			abort();
			return;
		}
		if (primalExitStatus == PrimalLoopExitStatus::DIDNT_STARTED) {
			// case when dual loop stopped on zero iteration
			// number of constraints is always > 0 (case without constraints considered separately) so INFEASIBILITY and FULL_ACTIVE_SET are impossible 
			// dual statuses 
			// the only possible dual status is ALL_DUAL_POSITIVE
			if (dualExitStatus == DualLoopExitStatus::ALL_DUAL_POSITIVE) {	
				assert(activeSetSize == 0);
			}
		} else {
			// dual loop made 1 or more iterations (no any return statements in primal loop)
			if (dualExitStatus == DualLoopExitStatus::INFEASIBILITY) {
				return;
			} 
		} 
		if (dualExitStatus == DualLoopExitStatus::ALL_DUAL_POSITIVE || dualExitStatus == DualLoopExitStatus::FULL_ACTIVE_SET) {
			if (solvePrimal()) {
				// solve primal for updated active set
				// if new active component is singular don't update primal
				primal = zp;
			}
		}
		// check feasibility
		if (origInfeasible()) {
			dualExitStatus = DualLoopExitStatus::INFEASIBILITY;
			return;
		} 
		getOrigSolution();
		unscaleD();
		cost = computeCost();
		if (st.logLevel > 2) {
			logger.message("before correction");
			checkConstraints();	
		}
		if (st.cholPvtStrategy == CholPivotingStrategy::FULL) {
			PTV(xOpt, pmtV);
			PermuteColumns(A, pmtV);
		}
		if (!checkFinalSolution()) {
			solverExitStatus = SolverExitStatus::CONSTRAINTS_VIOLATION;
			return;
		}
		if (st.logLevel > 2) {
			logger.message("after correction");
			checkConstraints();	
		}
		TimeIntervals tIntervals;
		timer->toIntervals(timer->TimeFromStart(), tIntervals);
		logger.message("total time: min", tIntervals.minutes, "sec", tIntervals.sec, "ms", tIntervals.ms, "mus", tIntervals.mus);
		logger.CloseFile();
		solverExitStatus = SolverExitStatus::SUCCESS;
	}

    bool NNLSQPSolver::checkFinalSolution() {
		//check bounds first
		if (isBounds) {
			const int nConstraints = this -> nConstraints - 2 * nVariables;
			const double factor = 1.0e-3;
			for (int i = 0; i < nVariables; ++i) {
				double diffUb = b[nConstraints + 2 * i] - xOpt[i];
				double diffLb = xOpt[i] + b[nConstraints + 2 * i + 1]; // if inside bounds -x <= b[i] => x + b[i] >= 0
                double dist = b[nConstraints + 2 * i] + b[nConstraints + 2 * i + 1];
				if (isSame(diffUb, 0.0) || (diffUb < 0.0 && diffUb >= -factor * dist)) {
					xOpt[i] = b[nConstraints + 2 * i];
				} else if (diffUb < -factor * dist) {
					return false;
				}
				if (isSame(diffLb, 0.0) || (diffLb < 0.0 && diffUb >= -factor * dist)) {
					xOpt[i] = -b[nConstraints + 2 * i + 1];
				} else if (diffLb < -factor * dist) {
					return false;
				} 
			}
		}
		Mult(A, xOpt, violations);
		for (int ic = 0; ic < nConstraints; ++ic) {
			double vl = violations[ic] - b[ic];
			if (vl > st.origFeasibilityTol) {
				return false;
			}
		}
		return true;
	}

	void NNLSQPSolver::Reset() {
		nVariables = 0;
		nConstraints = 0;
		nLEqConstraints = 0;
		nBounds = 0;
		minDualIndex = -1;
		gamma = 1.0;
		styGamma = 0.0;
		scaleFactor = 0.001;
		allActive = true;
		nextDualIteration = false;
		dualityGap = std::numeric_limits<double>::max();
		maxConstrViolation = std::numeric_limits<double>::min();
		activeSetSize = 0;
		dualExitStatus = DualLoopExitStatus::UNKNOWN;
		primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
		H.clear();
		A.clear();
		F.clear();
		activeSetIndices.clear();
		negativeZpIndices.clear();
        b.clear();
		d.clear();
		g.clear();
        primal.clear();
		dual.clear();
		zp.clear();
		vecS.clear();
		vecV.clear();
        lambdaOpt.clear();
		xOpt.clear();
		violations.clear();
		MtY.clear();
		pmtV.clear();
        matM.clear();
		cholFactor.clear();
        cholFactorInv.clear();
		singularConstraints.clear();
		lastIndices.clear();
		matrix_t matM; // M
		matrix_t cholFactor; // H = QT*Q
		matrix_t cholFactorInv; // Q^-1
		if (timer != nullptr) {
			timer -> Reset();
		}
		isBounds = false;
		cost = std::numeric_limits<double>::max();
  	}

	bool NNLSQPSolver::wasAddedRecently(int iConstraint) {
		int maxSize = 0;
		for (int i = 0; i < nConstraints; ++i) {
			if (dual[i] < 0.0) {
				++maxSize;
			}
		}
		maxSize = std::max(1, maxSize - static_cast<int>(singularConstraints.size()));

		if (lastIndices.size() >= maxSize) {
			lastIndices.clear();
			return false;
		}
		if (lastIndices.find(iConstraint) == lastIndices.end()) {
			return false;
		} else {
			return true;
		}
	}
////////////////////////////////////////////////////////////

	void LDL::Set(const matrix_t& A) {
		this->A = A;
		dimR = static_cast<int>(A.size());
		dimC = static_cast<int>(A.front().size());
		L.resize(dimR, std::vector<double>(dimR, 0.0));
		D.resize(dimR, 0.0);
		l.resize(dimR, 0.0);
		curIndex = 0;
		d = 0.0;

	}
	void LDL::Compute() {
		L.front().front() = 1.0;
		D.front() = getARowNormSquared(0);
		curIndex = 1;
		while(curIndex < dimR) {
			compute_l();
			compute_d();
			update_L();
			update_D();
			++curIndex;
		}
	}

	void LDL::Add(const std::vector<double>& row) {
		const int mSize = static_cast<int>(A.size());
		std::vector<double>b(mSize, 0.0); // b=A*rowT
		Mult(A, row, b);
		std::vector<double> l = b;
		solveLDb(b, l);
		double dd = DotProduct(row, row);
		for (int i = 0; i < mSize; ++i) {
			dd -= l[i] * D[i] * l[i];
		}
		for (auto& r : L) {
			r.resize(mSize + 1, 0.0);
		}
		L.push_back(l);
		L.back().push_back(1.0);
		D.push_back(dd);
		A.push_back(row);
	}

	void LDL::Remove(int i) {
		A.erase(A.begin() + i);
		// if remove last row
		if (i == A.size()) {
			L.resize(i);
			D.resize(i);
			for (int j = 0; j < i; ++j) {
				L[j].resize(i);
			}
			return;
		}
		const double dd = D[i];
		const int n = static_cast<int>(L.size());
		// i=0...n-1
		const int nRowsMdd = n - i - 1; //n-1,...,1
		const int nColsMdd = nRowsMdd + 1;
		matrix_t Mdd(nRowsMdd, std::vector<double>(nColsMdd, 0.0));
	   
		std::vector<double> droots(nRowsMdd);
		for (int j = 0; j < nRowsMdd; ++j) {
			double droot2 = 0.0;
			if (D[i + 1 + j] < 0.0) {
				std::cout << "LDL warning: " << "D[" << i + 1 + j << "]=" << D[1 + i + j] << "< 0" << std::endl;
				droot2 = 0.0;
			} else {
				droot2 = sqrt(D[i + 1 + j]);
			}
			droots[j] = droot2;
		}
		//fill L2*D2_1/2
		for (int ir = 0; ir < nRowsMdd; ++ir) {
			for (int ic = 0; ic < nRowsMdd; ++ic) {
				Mdd[ir][ic] = L[i + 1 + ir][i + 1 + ic] * droots[ic];
            }
		}
		//fill dd_1/2*L4
		if (dd < 0.0) {
			std::cout << "LDL warning: " << "D[" << i << "]=" << dd << "< 0" << std::endl;
		}
		const double ddSqrt = dd < 0.0 ? 0.0 : sqrt(dd);
		for (int ir = 0; ir < nRowsMdd; ++ir) {
			Mdd[ir][nRowsMdd] = ddSqrt * L[i + 1+ ir][i];
		}
		// solve L2_til * D2_til * L2_til
		LDL ldl;
		ldl.Set(Mdd);
		ldl.Compute();
		const matrix_t& Ltil = ldl.GetL();
		const std::vector<double>& Dtil = ldl.GetD();
		// update L,D with L_, D_
		update_L_remove(i, Ltil);
		D.resize(D.size() - 1);
		for (int j = 0; j < nRowsMdd; ++j) {
			D[j + i] = Dtil[j];
		}
	}

	void LDL::update_L_remove(int iRowDelete, const matrix_t& Ltil) {
		L.erase(L.begin() + iRowDelete);
		const int Lsize = static_cast<int>(L.size());
		for (int i = 0; i < Lsize; ++i) {
			L[i].resize(Lsize);
			if (i >= iRowDelete) {
				for (int j = 0; j < Lsize - iRowDelete; ++j) {
					L[i][j + iRowDelete] = Ltil[i - iRowDelete][j];
				}
			}
		}
	}

	const matrix_t& LDL::GetL() {
		return L;
	}
	const std::vector<double>& LDL::GetD() {
		return D;
	}

	void LDL::compute_l() {
		//L_i * D_i * l_i+1 = A1:i * A_i+1T
		//b = A1:i * A_i+1T
		std::vector<double> b(curIndex, 0.0);
		for (int i = 0; i < curIndex; ++i) {
			for (int j = 0; j < dimC; ++j) {
				b[i] += A[i][j] * A[curIndex][j];
			}
		}
		solveLDb(b, l);
	}
	void LDL::compute_d() {
		d = getARowNormSquared(curIndex);
		for (int i = 0; i < curIndex; ++i) {
			d -= l[i] * D[i] * l[i];
		}
	}
	void LDL::update_L() {
		L[curIndex][curIndex] = 1.0;
		for (int i = 0; i < curIndex; ++i) {
			L[curIndex][i] = l[i];
		}
	}
	void LDL::update_D() {
		if (d <= 0.0) {
			std::cout << "LDL warning: " << "d=" << d << "<0" << std::endl;
			d = 0.0;
		} 
		D[curIndex] = d;
	}
	double LDL::getARowNormSquared(int row) const {
		double norm2 = 0;
		for (int i = 0; i < dimC; ++i) {
			norm2 += A[row][i] * A[row][i];
		}
		return norm2;
	}
	void LDL::solveLDb(const std::vector<double>& b, std::vector<double>& l) {
		const int n = b.size();
		for (int i = 0; i < n; ++i) {
			if (std::fabs(D[i]) < 1.0e-20) {
				l[i] = 0.0;
			} else {
				double sum = 0;
				for (int j = 0; j < i; ++j) {
					sum += L[i][j] * D[j] * l[j];
				}
				l[i] = (b[i] - sum) / D[i]; // (b[i] - sum) / L[i][i] * D[i] , L[i][i] = 1 
			}
		}
	}
	int MMTbSolver::Solve(const matrix_t& M, const std::vector<double>& b) {
		//solve MMTx=b
		assert(M.size() == b.size());
		LDL ldl;
		forward.resize(M.size());
		backward.resize(M.size());
		ldl.Set(M);
		ldl.Compute();
		ndzero = 0;
		std::vector<int> dzeroIndices(M.size(), -1);
		int j = 0;
		for (int i = 0; i < M.size(); ++i) {
			if (std::fabs(ldl.GetD()[i]) < zeroTol) {
				ndzero += 1;
				dzeroIndices[j++] = i;
			}
		}
		if (ndzero >= 0) {
			SolveForward(ldl.GetL(), b);
			SolveBackward(ldl.GetD(), ldl.GetL());
	    } else {
			GetMMTKernel(dzeroIndices, ldl.GetL(), backward);
		}
		return ndzero;
	}
	void MMTbSolver::SolveForward(const matrix_t& L, const std::vector<double>& b) {
		const int n = b.size();
		for (int i = 0; i < n; ++i) {
			double sum = 0.0;
			for (int j = 0; j < i; ++j) {
				sum += L[i][j] * forward[j];
			}
			forward[i] = b[i] - sum;
		}
	}
	void MMTbSolver::SolveBackward(const std::vector<double>& D, const matrix_t& L) {
		const int n = forward.size();
		for (int i = n - 1; i >= 0; --i) {
			double sum = 0.0;
			for (int j = i + 1; j < n; ++j) {
				sum += L[j][i] * D[i] * backward[j];
			}
			backward[i] = std::fabs(D[i]) < zeroTol ? 0.0 : (forward[i] - sum) / D[i];
		}
	}
	void MMTbSolver::GetMMTKernel(const std::vector<int>& dzeroIndices, const matrix_t& L, std::vector<double>& ker) {

		if (ndzero > 0) { // last element
			const int zeroIndex = dzeroIndices.front();
			std::fill(ker.begin(), ker.end(), 0.0);
			const auto& row = L[zeroIndex];
			//solve backward Lx = -l
			for (int i = zeroIndex - 1; i >= 0; --i) {
				double sum = 0.0;
				for (int j = i + 1; j < zeroIndex; ++j) {
					sum += L[j][i] * ker[j];
				}
				ker[i] = -row[i] - sum;
			}
			ker[zeroIndex] = 1.0;
		}

	}
	const std::vector<double>& MMTbSolver::GetSolution() {
		return backward;
	}
	int MMTbSolver::nDZero() {
		return ndzero;
	}
    DBScaler::DBScaler(DBScalerStrategy strategy):
		scaleStrategy(strategy)
	{}
	double DBScaler::Scale(const matrix_t& M, const std::vector<double>& s, const NNLSQPSolver::Settings& solverSettings) {
	    // on zero dual iteration when active set is empty yet dual = gamma * s
		// if any component of s is very small, it's comparison with tolerance in active set selection procedure 
		// may be incorrect which leads to incorrect new active component 
		// don't scale to very small s component 
		double maxSComponent = 0.0;
		double minSComponent = std::numeric_limits<double>::max();
		for (const auto& sc : s) {
			const double absSc = std::fabs(sc);
			if (absSc > maxSComponent) {
				maxSComponent = absSc;
			}
			if (absSc < minSComponent && absSc > 0.0) {
				minSComponent = absSc;
			}
		}
		if (maxSComponent == 0.0) {
            return 1.0;   //TO DO in this case vector S is zero, incorrect input
		}
		double maxNorm = 0.0;
		double minNorm = std::numeric_limits<double>::max();
		double maxEl = 0.0;
		double minEl = std::numeric_limits<double>::max();
		for (const auto& row : M) {
			for (const auto& el: row) {
				const double absEl = std::fabs(el);
				if (absEl > maxEl) {
					maxEl = absEl;
				}
				if (absEl < minEl && absEl > 0.0) {
					minEl = absEl;
				}
			}
			double norm = sqrt(DotProduct(row, row));
			if (norm > maxNorm) {
				maxNorm = norm;
			}
			if (norm < minNorm) {
				minNorm = norm;
			}
		}
		assert(!isSame(minNorm, 0.0));
		double balanceFactor_1 =  maxSComponent / minNorm;
		double balanceFactor_2 =  maxSComponent / maxNorm;
		if (scaleStrategy == DBScalerStrategy::SCALE_FACTOR) {
			// balance to fixed ratio maxBalanceFactor
			if (balanceFactor_1 <= balanceUpperBound && balanceFactor_1 >= balanceLowerBound) {
				// normal case, normalize to maxBalanceFactor
				if (balanceFactor_1 > maxBalanceFactor) {
					// s[i] >> ||M[i]||
					// dual[i] > (gamma + s_T * y) * origFeasibilityTol => (A * x)[i] - b[i] <= origFeasibilityTol
					// Increasing gamma from iteration to iteration improves numerical stability 
					// but for zero iteration if gamma == 1 |s[i]_min| must be > origFeasibilityTol
					const double unscaledTol = solverSettings.origFeasibilityTol; 
				    double scaleCoef = maxBalanceFactor / balanceFactor_1;

					while (unscaledTol * scaleCoef < 1.0e-14 ) {
						scaleCoef *= 10.0;
					}
					const_cast<double&>(solverSettings.origFeasibilityTol) = unscaledTol * scaleCoef; 
				
					return scaleCoef;
				} else if ( 1.0 / balanceFactor_1  > maxBalanceFactor) {
					// ||M[i]|| >> s[i]
					return balanceFactor_1 / maxBalanceFactor;
				} else {
					return 1.0;
				}
			} else if (balanceFactor_1 > balanceUpperBound ) {
				// extremely unbalanced matrix, s dominates
				return extremeFactorS;
			} else {
				// extremely unbalanced matrix, M dominates
				return extremeFactorM;
			}
		} else if (scaleStrategy == DBScalerStrategy::BALANCE) {
			// compute balance factor
			if (maxEl == 0.0) {
				return 1.0; //TO DO in this case matrix M is zero, incorrect input
			}
			double mBalanceFactor = maxEl / minEl;
			double sBalanceFactor = maxSComponent / minSComponent;

			

		}
	}

}
