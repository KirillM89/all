#include "NNLSQPSolver.h"
#include <cmath>
#include <algorithm>
namespace QP_NNLS {
Core::Core():
    nVariables(0),
    nConstraints(0),
    nEqConstraints(0),
    newActiveIndex(0),
    rptInterval(0),
    singularIndex(0),
    dualExitStatus(DualLoopExitStatus::UNKNOWN),
    primalExitStatus(PrimalLoopExitStatus::UNKNOWN),
    gamma(0),
    styGamma(0),
    scaleFactorDB(1.0),
    rsNorm(0),
    newActive(0),
    dualTolerance(std::numeric_limits<double>::min()),
    dualityGap(std::numeric_limits<double>::max()),
    cost(std::numeric_limits<double>::max()),
    dbScaler(nullptr),
    timer(std::make_unique<wcTimer>()),
    uCallback(std::make_unique<Callback>())
{
    ws.Clear();
    SetDefaultSettings();
}
void Core::WorkSpace::Clear() {
    s.clear();
    zp.clear();
    primal.clear();
    dual.clear();
    lambda.clear();
    MTY.clear();
    x.clear();
    c.clear();
    b.clear();
    activeConstraints.clear();
    negativeZp.clear();
    v.clear();
    slack.clear();
    pmt.clear();
    H.clear();
    M.clear();
    Jac.clear();
    Chol.clear();
    CholInv.clear();
    addHistory = {};
}
void Core::SetDefaultSettings() {
    settings = CoreSettings();
}
void Core::ResetProblem() {
    ws.Clear();
}
void Core::Set(const CoreSettings& settings) {
    this->settings = settings;
}
void Core::SetCallback(std::unique_ptr<Callback> callback) {
    if (callback != nullptr) {
        uCallback = std::move(callback);
    }
}
bool Core::InitProblem(const DenseQPProblem &problem) {
    bool stat = PrepareNNLS(problem);
    uCallback->initData.Chol = ws.Chol;
    uCallback->initData.CholInv = ws.CholInv;
    uCallback->initData.M = ws.M;
    uCallback->initData.s = ws.s;
    uCallback->initData.c = ws.c;
    uCallback->initData.b = ws.b;
    uCallback -> ProcessData(1);
    return stat;
}
void Core::AllocateWs() {
    ws.primal.resize(nConstraints, 0.0);
    ws.dual.resize(nConstraints, 0.0);
    ws.v.resize(nVariables, 0.0);
    ws.zp.resize(nConstraints, 0.0);
    ws.lambda.resize(nConstraints, 0.0);
    ws.x.resize(nVariables, 0.0);
    ws.MTY.resize(nVariables, 0.0);
    ws.slack.resize(nConstraints, 0.0);
    ws.Chol.resize(nVariables, std::vector<double>(nVariables, 0.0));
    ws.CholInv.resize(nVariables, std::vector<double>(nVariables, 0.0));
    ws.M.resize(nConstraints, std::vector<double>(nVariables, 0.0));
    ws.activeConstraints.clear();
    ws.addHistory.clear();
}
void Core::ExtendJacobian(const matrix_t& Jac, const std::vector<double>& b,
                          const std::vector<double>& lb, const std::vector<double>& ub) {
    ws.Jac = Jac;
    ws.Jac.resize(ws.Jac.size() + 2 * nConstraints, std::vector<double>(nVariables, 0.0));
    ws.b = b;
    ws.b.resize(ws.b.size() + 2 * nConstraints, 0.0);
    for (unsg_t i = 0; i <  nVariables; ++i) {
        const int ibg = 2 * i + nConstraints;
        ws.Jac[ibg][i] = 1.0;
        ws.Jac[ibg + 1][i] = -1.0;
        ws.b[ibg] = ub[i];
        ws.b[ibg + 1] = -lb[i];
    }
    nConstraints += 2 * nVariables;
}
bool Core::PrepareNNLS(const DenseQPProblem &problem) {
    nVariables = static_cast<unsg_t>(problem.H.size());
    nConstraints = static_cast<unsg_t>(problem.A.size());
    nEqConstraints = problem.nEqConstraints;
    ws.H = problem.H;
    ws.c = problem.c;
    ExtendJacobian(problem.A, problem.b, problem.lw, problem.up);
    AllocateWs();
    dbScaler =  std::make_unique<DBScaler>(settings.dbScalerStrategy);
    timer->Start();
    if (settings.cholPvtStrategy == CholPivotingStrategy::NO_PIVOTING) {
        CholetskyOutput cholOutput;
        if(!ComputeCholFactorT(problem.H, ws.Chol, cholOutput)) {   // H = L_T * L
            uCallback->initData.InitStatus = InitStageStatus::CHOLETSKY;
            return false;
        }
    } else if (settings.cholPvtStrategy == CholPivotingStrategy::FULL) {
        ws.pmt.resize(nVariables, -1.0);
        // full pivoting:
        // x == P * x_n;  P - permuation matrix
        // 0.5 * x_T * H * x + c * x = 0.5 * x_n_T * P_T * H * P * x_n + c_T * P * x_n = 0.5 * x_n_T * H_n * x_n + c_n_T * x_n
        // H_n = P_T * H * P ; c_n = P_T * c
        // A * x <= b  A * P *x_n <= b  A_n = A * P   A_n * x_n <= b
        if (ComputeCholFactorTFullPivoting(ws.H, ws.Chol, ws.pmt) != 0) { // H -> H_n
            uCallback->initData.InitStatus = InitStageStatus::CHOLETSKY;
            return false;
        }
        PermuteColumns(ws.Jac, ws.pmt);
        PTV(ws.c, ws.pmt);
    }
    TimePoint(uCallback -> initData.tChol);
    InvertTriangle(ws.Chol, ws.CholInv);   // Q^-1
    TimePoint(uCallback -> initData.tInv);
    Mult(ws.Jac, ws.Chol, ws.M);           // M = A * Q^-1   nConstraints x nVariables
    TimePoint(uCallback -> initData.tM);
    MultTransp(ws.CholInv, ws.c, ws.v);    // v = Q^-T * d nVariables
    std::vector<double> MByV(nConstraints);
    Mult(ws.M, ws.v, MByV);                // M * v nConstraints
    VSum(MByV, ws.b, ws.s);
    scaleFactorDB = dbScaler -> Scale(ws.M, ws.s, settings.origPrimalFsb);
    ScaleD();
}
void Core::TimePoint(std::string& buf) {
    TimeIntervals tIntervals;
    timer->toIntervals(timer->Ticks(), tIntervals);
    buf = std::to_string(tIntervals.minutes) + " min " +
          std::to_string(tIntervals.sec) + " sec " +
          std::to_string(tIntervals.ms) + " ms " +
          std::to_string(tIntervals.mus) + " mus";
}

bool Core::OrigInfeasible() {
    MultTransp(ws.M, ws.primal, ws.activeConstraints, ws.MTY); // M_T * primal
    styGamma = gamma + DotProduct(ws.s, ws.primal, ws.activeConstraints);
    rsNorm = DotProduct(ws.MTY, ws.MTY) + styGamma * styGamma;
    return rsNorm < settings.nnlsResidNormFsb;
}

bool Core::FullActiveSet() {
    return (static_cast<unsg_t>(ws.activeConstraints.size()) == nConstraints);
}
void Core::ComputeDualVariable() {
    Mult(ws.M, ws.MTY, ws.dual); // M * M_T * primal
    for (unsg_t i = 0; i < nConstraints; ++i) {
        ws.dual[i] += styGamma * ws.s[i];
    }
    // correction of dual variables
    for (auto& indx : ws.activeConstraints) {
        ws.dual[indx] = 0.0;
    }
    styGamma = gamma + DotProduct(ws.s, ws.primal);
}
bool Core::SkipCandidate(unsg_t indx) {
    if (rptInterval > 0) {
        return (std::find(ws.addHistory.begin(), ws.addHistory.end(), indx) != ws.addHistory.end());
    } else if (!settings.actSetUpdtSettings.rejectSingular && singularIndex == indx) {
        return true;
    } else {
        return false;
    }
}
void Core::AddToActiveSet(unsg_t indx) {
    ws.activeConstraints.insert(indx);
    if (rptInterval > 0) { // if rptInterval > 0 use history saving
        if(ws.addHistory.size() >= rptInterval) {
            ws.addHistory.pop_front();
        }
        ws.addHistory.push_back(indx);
    }
}
void Core::RmvFromActiveSet(unsg_t indx) {
    ws.activeConstraints.erase(indx);
}

bool Core::IsCandidateForNewActive(unsg_t indx, double toCompare) {
    bool res = false;
    const double dl = ws.dual[indx];
    if (!SkipCandidate(indx) && (dl < dualTolerance && dl < toCompare)) {
        newActiveIndex = indx;
        res = true;
    }
    return res;
}
unsg_t Core::SelectNewActiveComponent() {
    // strategy with selection of minimum dual component as new active
    double newActive = std::numeric_limits<double>::max();
    newActiveIndex = nConstraints; //default value
    bool newFound = false;
    if (settings.actSetUpdtSettings.firstInactive) {
        // first check inactive components
        for (unsg_t i = 0; i < nConstraints; ++i) {
            if ((ws.activeConstraints.find(i) == ws.activeConstraints.end()) &&
                IsCandidateForNewActive(i, newActive)) {
                newActive = ws.dual[i];
                newFound = true;
            }
        }
        if (!newFound) {
            //first check active components. TODO : May dual be negative ???
            for (auto i : ws.activeConstraints) {
                if (IsCandidateForNewActive(i, newActive)) {
                    newActive = ws.dual[i];
                    newFound = true;
                }
            }
        }
    } else {
        for (unsg_t i = 0; i < nConstraints; ++i) {
            if (IsCandidateForNewActive(i, newActive)) {
                newActive = ws.dual[i];
                newFound = true;
            }
        }
    }
    if (!newFound) { //finally check in indices to Skip
        //TODO
    }
    return newActiveIndex;
}

unsg_t Core::SolvePrimal() {
    const std::size_t nActive = ws.activeConstraints.size();
    matrix_t M{};
    std::vector<double> s{};
    // [M s] * [M_T / s_T] = -gamma * s
    if (nActive > 0) {
        M.resize(nActive);
        s.resize(nActive);
        std::size_t i = 0;
        for (const unsg_t& indx: ws.activeConstraints ) {
            M[i] = ws.M[indx];
            s[i]= -gamma * ws.s[indx];
            M.back().push_back(ws.s[indx]);
            ++i;
        }
    } else {
        M.resize(nConstraints);
        s.resize(nConstraints);
        for (int i = 0; i < nConstraints; ++i) {
            M[i] = {ws.s[i]};
            s[i] = -gamma * ws.s[i];
        }
    }
    MMTbSolver mmtb;
    int nDNegative = mmtb.Solve(M, s);
    if (nDNegative > 0 && !settings.actSetUpdtSettings.rejectSingular) {
        const auto& sol= mmtb.GetSolution();
        std::fill(ws.zp.begin(), ws.zp.end(), 0.0);
        ws.negativeZp.clear();
        std::size_t i = 0;
        for (auto indx: ws.activeConstraints) {
            ws.zp[indx] = sol[i];
            if (sol[i] < settings.nnlsPrimalZero) {
                ws.negativeZp.insert(indx);
            }
        }
    }
    // TODO: check quality
    return nDNegative;
}

bool Core::MakeLineSearch() {
    double minStep = std::numeric_limits<double>::max();
    bool stepFound = false;
    // case if all zp are non-negative must be proccessed before this function, negativePrimalIndices must not be empty
    for (auto indx: ws.negativeZp) {
        double denominator = ws.primal[indx] - ws.zp[indx];
        if (!isSame(denominator, 0.0)) {
            minStep = std::fmin(minStep, ws.primal[indx] / denominator);
            stepFound = true;
        }
    }
    if (stepFound) {
        //primal_next = primal + step * (zp - primal)
        double gammaCorrection = 0.0;
        for (unsg_t i = 0; i < nConstraints; ++i) {
            ws.primal[i] += minStep * (ws.zp[i] - ws.primal[i]);
            if (std::fabs(ws.primal[i]) < settings.nnlsPrimalZero) {
                gammaCorrection += std::fabs(ws.s[i]);
                ws.activeConstraints.erase(i);
            }
        }
        //UpdateGammaAfterLineSearch();
        if (settings.minNNLSDualTol < 1.0e-10) {
            gamma = std::fabs(gamma - gammaCorrection);
        }
    }
    return stepFound || ws.negativeZp.empty();
}

int Core::UpdatePrimal() {
    int res = 0;
    if (SolvePrimal() > 0) {
        res |= SINGULARITY;
    }
    if (!settings.actSetUpdtSettings.rejectSingular) {
        if (!ws.negativeZp.empty() ) {
            if (!MakeLineSearch()) {
                res |= LINE_SEARCH_FAILED;
            }
        } else {
            ws.primal = ws.zp;
        }
    }
    return res;
}

void Core::ScaleD() {
    //dbScaler = std::make_unique<DBScaler>(st.dbScalerStrategy);
    //scaleFactorDB = dbScaler -> Scale(ws.M, ws.s, st);
    const unsg_t n = std::max(nConstraints, nVariables);
    for (unsg_t i = 0; i < n; ++i) {
        if (i < nConstraints) {
            ws.b[i] *= scaleFactorDB;
            ws.s[i] *= scaleFactorDB;
        }
        if (i < nVariables) {
            ws.c[i] *= scaleFactorDB;
            ws.v[i] *= scaleFactorDB;
        }
    }
}

void Core::UpdateGammaOnDualIteration() {

}
void Core::ComputeCost() {

}
void Core::ComputeExactLambdaOnActiveSet() {
    // Correct lambdas for active constraints to improve feasibility
    matrix_t M;
    std::vector<double> s;
    for (auto i :ws.activeConstraints) {
        M.push_back(ws.M[i]);
        s.push_back(ws.s[i]);
    }
    if (M.empty()) {
        return;
    }
    MMTbSolver mmtb;
    mmtb.Solve(M, s);
    std::vector<double> solActiveSet = mmtb.GetSolution();
    std::size_t ii = 0;
    for (auto i :ws.activeConstraints) {
        ws.lambda[i] = solActiveSet[ii++];
    }
}

double Core::ComputeDualityGap() {
    // x, lambda must be correct!
    // For original problem
    // A * x_opt - b = -s - M * M_T * lambda
    // Compute -s - M * M_T * lambda
    std::vector<double> MMTL(nConstraints);
    std::vector<double> violations(nConstraints);
    MultTransp(ws.M, ws.lambda, ws.MTY);
    Mult(ws.M, ws.MTY, MMTL);
    VSum(MMTL, ws.s, violations);
    const double lamTByS = DotProduct(ws.lambda, ws.s);
    const double vTv = DotProduct(ws.v, ws.v);
    const double mty2 = DotProduct(ws.MTY, ws.MTY);
    const double dualValue = -0.5 * (mty2 + vTv) - lamTByS;
    std::vector<double> Ax(nConstraints);
    Mult(ws.Jac, ws.x, Ax);
    const double fsb = DotProduct(Ax, ws.lambda) - DotProduct(ws.b, ws.lambda);
    ComputeCost();
    dualityGap = cost + fsb - dualValue;
}

void Core::ComputeOrigSolution() {
    double sty = DotProduct(ws.s, ws.primal, ws.activeConstraints);
    double lambdaTerm = -1.0 / (gamma + sty);
    for (unsg_t i = 0; i < nConstraints; ++i) {
        ws.lambda[i] = lambdaTerm * ws.primal[i] ;
    }
    ComputeExactLambdaOnActiveSet();
    std::vector<double> u(nVariables, 0.0);
    MultTransp(ws.M, ws.lambda, ws.activeConstraints, u);
    std::vector<double> u_v(nVariables);
    for (unsg_t i = 0; i < nVariables; ++i) {
        u_v[i] = u[i] - ws.v[i];
    }
    Mult(ws.CholInv, u_v, ws.x);
    for (unsg_t i = 0; i < nConstraints; ++i) {
        ws.lambda[i] *= -1.0;
    }
}

void Core::FillOutput() {

}

void Core::Solve() {
    dualExitStatus = DualLoopExitStatus::UNKNOWN;
    primalExitStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    unsg_t dualIteration = 0;
    gamma = 1.0;
    while (dualIteration < settings.nDualIterations) {
        if (OrigInfeasible()) {
            dualExitStatus = DualLoopExitStatus::INFEASIBILITY;
            break;
        }
        if (FullActiveSet()) {
            dualExitStatus = DualLoopExitStatus::FULL_ACTIVE_SET;
            break;
        }
        ComputeDualVariable();
        dualTolerance = -styGamma * settings.origPrimalFsb; // primal feasiblility was scaled in DB scaling
        SelectNewActiveComponent();
        if(newActiveIndex == nConstraints) { //set to nConstraints in not found
            dualExitStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
            break;
        }
        AddToActiveSet(newActiveIndex);
        UpdateGammaOnDualIteration();
        unsg_t primalIteration = 0;
        primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
        singularIndex = nConstraints;
        while (primalIteration < settings.nPrimalIterations) {
            if (ws.activeConstraints.empty()) {
                primalExitStatus = primalIteration == 0 ? PrimalLoopExitStatus::EMPTY_ACTIVE_SET_ON_ZERO_ITERATION :
                                                          PrimalLoopExitStatus::EMPTY_ACTIVE_SET;
                break;
            }
            int prStat = UpdatePrimal();
            const bool success = (prStat == 0) ||((prStat == SINGULARITY)
                    && !settings.actSetUpdtSettings.rejectSingular);
            const bool rejectSingular = (prStat == SINGULARITY) && settings.actSetUpdtSettings.rejectSingular;
            if (success) {
                if (ws.negativeZp.empty()) {
                    primalExitStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
                    break;
                } else {
                    ++primalIteration;
                }
            } else if (rejectSingular) {
                primalExitStatus = PrimalLoopExitStatus::SINGULAR_MATRIX;
                RmvFromActiveSet(newActiveIndex);
                singularIndex = newActiveIndex; // save singular index
                break;
            } else {
                primalExitStatus = PrimalLoopExitStatus::LINE_SEARCH_FAILED;
                break;
            }
        }
        if (primalIteration >= settings.nPrimalIterations) {
            primalExitStatus = PrimalLoopExitStatus::ITERATIONS;
        }
    }
    if (dualIteration >= settings.nDualIterations) {
        dualExitStatus = DualLoopExitStatus::ITERATIONS;
    }
    if (dualExitStatus != DualLoopExitStatus::INFEASIBILITY) {
        ComputeOrigSolution();
    }
    FillOutput();
}

}
