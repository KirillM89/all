#include "NNLSQPSolver.h"
#include <cmath>
namespace QP_NNLS {
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
void Core::ResetProblem() {
    ws.Clear();
}
void Core::Set(const CoreSettings& settings) {
    this->settings = settings;
}
void Core::SetCallback(std::shared_ptr<Callback> callback) {
    uCallback = callback;
}
bool Core::InitProblem(const DenseQPProblem &problem) {
    timer = std::make_unique<wcTimer>();
    bool stat = PrepareNNLS(problem);
    uCallback -> ProcessData();
    return stat;
}
bool Core::PrepareNNLS(const DenseQPProblem &problem) {
    nVariables = static_cast<unsg_t>(problem.H.size());
    nConstraints = static_cast<unsg_t>(problem.A.size());
    nEqConstraints = problem.nEqConstraints;
    ws.H = problem.H;
    ws.c = problem.c;
    // Extend Jacobian
    nConstraints += 2 * nVariables;
    ws.Jac = problem.A;
    ws.Jac.resize(nConstraints, std::vector<double>(nVariables, 0.0));
    ws.b = problem.b;
    ws.b.resize(nConstraints, 0.0);
    // Allocate ws
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
    // Compute NNLS
    timer->Start();
    if (settings.cholPvtStrategy == CholPivotingStrategy::NO_PIVOTING) {
        CholetskyOutput cholOutput;
        if(!ComputeCholFactorT(problem.H, ws.Chol, cholOutput)) {   // H = L_T * L
            if (uCallback != nullptr) {
                uCallback->initData.InitStatus = InitStageStatus::CHOLETSKY;
            }
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
            if (uCallback != nullptr) {
                uCallback->initData.InitStatus = InitStageStatus::CHOLETSKY;
            }
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
    ScaleD();
    // Fill log data
    if (uCallback != nullptr) {
        uCallback->initData.Chol = ws.Chol;
        uCallback->initData.CholInv = ws.CholInv;
        uCallback->initData.M = ws.M;
        uCallback->initData.s = ws.s;
        uCallback->initData.c = ws.c;
        uCallback->initData.b = ws.b;
    }
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

}
void Core::AddToActiveSet(unsg_t indx) {

}
void Core::RmvFromActiveSet(unsg_t indx) {

}
void Core::ResetPrimal(){

}
unsg_t Core::SelectNewActiveComponent() {
    // strategy with selection of minimum dual component as new active
    double minDual = std::numeric_limits<double>::max();
    bool newFound = false;
    if (settings.actSetUpdtSettings.firstInactive) {
        // first check inactive components
        for (unsg_t i = 0; i < nConstraints; ++i) {
            if (ws.activeConstraints.find(i) != ws.activeConstraints.end() || SkipCandidate(i)) {
                continue;
            }
            const double dl = ws.dual[i];
            if (dl < settings.minNNLSDualTol && dl < minDual) {
                minDual = dl;
                newFound = true;
                newActive = minDual;
                newActiveIndex = i;
            }
        }
        if (!newFound) {
            //first check active components. TODO : May dual be negative ???
            for (auto indx : ws.activeConstraints) {
                if (!SkipCandidate(indx)) {

                }
            }
        }
    }
}

Core::PrimalRetStatus Core::SolvePrimal() {
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
    // TODO: check quality
    if (nDNegative > 0) {
        return PrimalRetStatus::SINGULARITY;
    }
    return PrimalRetStatus::SUCCESS;
}

void Core::MakeLineSearch() {
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
    //primal_next = primal + step * (zp - primal)
    double gammaCorrection = 0.0;
    for (unsg_t i = 0; i < nConstraints; ++i) {
        ws.primal[i] += minStep * (ws.zp[i] - ws.primal[i]);
        if (std::fabs(ws.primal[i]) < settings.nnlsPrimalZero) {
            gammaCorrection += std::fabs(ws.s[i]);
            ws.activeConstraints.erase(i);
        }
    }
    if (settings.minNNLSDualTol < 1.0e-10) {
        gamma = std::fabs(gamma - gammaCorrection);
    }
}

Core::PrimalRetStatus Core::UpdatePrimal() {
    PrimalRetStatus res = SolvePrimal();
    if (res == PrimalRetStatus::SINGULARITY &&
        settings.actSetUpdtSettings.rejectSingular) {
    } else if (!ws.negativeZp.empty()) {
        MakeLineSearch();
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
void Core::Solve() {
    dualExitStatus = DualLoopExitStatus::UNKNOWN;
    primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
    unsg_t dualIteration = 0;
    double dualTolerance = std::numeric_limits<double>::max();
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
        unsg_t newActive = SelectNewActiveComponent();
        if(newActive == nConstraints) { //set to nConstraints in not found
            dualExitStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
            break;
        }
        UpdateGammaOnDualIteration();
        unsg_t primalIteration = 0;
        primalExitStatus = PrimalLoopExitStatus::UNKNOWN;
        while (primalIteration < settings.nPrimalIterations) {
            if (ws.activeConstraints.empty()) {
                primalExitStatus = primalIteration == 0 ? PrimalLoopExitStatus::EMPTY_ACTIVE_SET_ON_ZERO_ITERATION :
                                                          PrimalLoopExitStatus::EMPTY_ACTIVE_SET;
                break;
            }
            PrimalRetStatus prStat = UpdatePrimal();
        }

    }

}
}
