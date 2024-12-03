#include "NNLSQPSolver.h"
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
    timer->Start();
    //dbScaler = std::make_unique<DBScaler>(st.dbScalerStrategy);
    if (!PrepareNNLS(problem)) {
        return false;
    }
    //uCallback.data = ...
    //uCallback -> processResults();
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
            //uCallback.data
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
}

void Core::TimePoint(std::string& buf) {

}

void Core::ScaleD() {
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
}
