#include "qp.h"
#include "qld.h"
#include <cassert>
#include <cmath>
#include <gtest/gtest.h>

namespace QP_SOLVERS {
    
    std::vector<double> FlattenByColumns(const matrix_t& matrix) {
        size_t m = matrix.size();
        size_t n = m ? matrix[0].size() : 0;
        std::vector<double> matrixByColumns(m * n);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                matrixByColumns[m * j + i] = matrix[i][j];
            }
        }
        return matrixByColumns;
    }
    void solveQLD(const QPInput& input, QPOutput& output){
        int nMax = static_cast<int>(input.m_H.size());
        int nConstraints = static_cast<int>(input.m_A.size());
        int nEqConstraints = 0;
        int nVariables = static_cast<int>(input.m_x0.size());
        int mRows = std::max(nConstraints, 1);
        int mPlus2N = nConstraints + 2 * nVariables;
        int iOut = 6;
        int zero = 0;
        int status = 0;
        int intArraySize = nVariables;
        int workArraySize = static_cast<int>(1.5 * nMax  * nMax)  + 10 * nMax  + 2 * mRows + 1;
        std::vector<int> intArray(nVariables, 0.0);
        std::vector<double> workArray(workArraySize);
        matrix_t m_A_QLD = input.m_A;
        for (int i = 0; i < nConstraints; ++i) {
            for (int j = 0; j < nVariables; ++j) {
                m_A_QLD[i][j] = -input.m_A[i][j];
            }
        }
        std::vector<double> hFlat = FlattenByColumns(input.m_H);
        std::vector<double> aFlat = FlattenByColumns(m_A_QLD);
        std::vector<double> lambda(mPlus2N, 0.0);
        // The following asks QP-solver to use LDL-decomposition (square-root-free Cholesky decomposition)
        // C = L*D*L^T, where L - lower triangular, D - diagonal
        intArray[0] = 1;  // use LDL-decomposition
        const double machinePrecision = 1e-16;
        std::vector<double> x = input.m_x0;
        ql0001_(&nConstraints, &nEqConstraints, &mRows, &nVariables,
                &nMax, &mPlus2N, hFlat.data(), input.m_cV.data(), aFlat.data(), input.m_bV.data(), input.m_lower.data(), input.m_upper.data(),
                x.data(), lambda.data(), &iOut, &status, &zero, workArray.data(), &workArraySize, intArray.data(),
                &intArraySize, &machinePrecision);

        double cost = 0.0;
        const double half = 0.5;
        for (std::size_t i = 0; i < nVariables; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                cost += input.m_H[i][j] * x[i] * x[j] * (i == j ? half : 1.0);
            }
            cost += input.m_cV[i] * x[i];
        }
        const std::string qpReturnMessages[7] = {"", "Termination after too many iterations: 40*(N+M)",
        "Insufficient accuracy", "Inconsistency, division by zero", "Numerical instability",
        "Incorrect epsilon or working/int array sizes", "Inconsistent constraint"};
        if (status > 100) {
            output.m_errMsg = "QP: " + qpReturnMessages[6] + std::to_string(status - 100);
        } else if (status && status < 6) {
            output.m_errMsg = "QP: " + qpReturnMessages[status];
        }
     
        output.exitStatus = status == 0 ? 0 : 1;
        std::cout << "QLD status " << status << std::endl;
        if (status == 0) {
            output.m_x = x;
            output.m_cost = cost;
            output.m_lambda = std::vector<double>(lambda.begin(), lambda.begin() + nConstraints);
            output.m_lambdaL = std::vector<double>(lambda.begin() + nConstraints, lambda.begin() + nConstraints + nVariables);
            output.m_lambdaU = std::vector<double>(lambda.begin() + nConstraints + nVariables, lambda.end());
        }
    } 

#ifdef DAQP
void solveDAQP(const QPInput& input, QPOutput& output) {
    std::size_t nVariables = input.m_x0.size();
    std::size_t nConstraints = input.m_A.size();
    std::size_t nEqConstraints = 0;

    std::vector<double> hessFlatten(nVariables * nVariables, 0.0);
    for (std::size_t i = 0; i < nVariables ; i++) {
        for (std::size_t j = 0; j < nVariables ; j++) {
            hessFlatten[i * nVariables + j] = input.m_H[i][j];
        }
    }
    std::vector<double> jacFlatten(nConstraints * nVariables, 0.0);
    for (std::size_t i = 0; i < nConstraints; i++) {
        for (std::size_t j = 0; j < nVariables; j++) {
            jacFlatten[i * nVariables + j] = input.m_A[i][j];
            if(std::fabs(jacFlatten[i * nVariables + j]) < 1e-12) {
                jacFlatten[i * nVariables + j] = 1e-12;
            }
        }
    }
    const double defaultBoundValue = 1.0e19;
    bool boundsAsConstraints = true;
    std::vector<double> lower;
    std::vector<double> upper;
    int nSimpleConstraints = 0;
    if (boundsAsConstraints) {
        const std::size_t jacSize = nConstraints * nVariables + nVariables * nVariables;
        jacFlatten.resize(jacSize, 0.0);
        lower.resize(nConstraints + nVariables, 0.0);
        upper.resize(nConstraints + nVariables, 0.0);
        for (std::size_t i = 0; i < nConstraints; i++) {
            lower[i] = -input.m_bV[i];
            upper[i] = defaultBoundValue;
        }
        for (std::size_t i = 0; i < nVariables; ++i) {
            jacFlatten[(nConstraints + i) * nVariables  + i] = 1.0;
            lower[i + nConstraints] = input.m_lower[i];
            upper[i + nConstraints] = input.m_upper[i];
        } 
    } else {

        lower = input.m_lower;
        upper = input.m_upper;
    
        lower.resize(nConstraints + nVariables, -defaultBoundValue);
        upper.resize(nConstraints + nVariables, defaultBoundValue);
        for (std::size_t i = 0; i < nConstraints; i++) {
             lower[i + nVariables] = -input.m_bV[i];
        }
        nSimpleConstraints = nVariables;
    }
    std::vector<int> constraintTypes(nConstraints + nVariables, 0); 
    DAQPSettings settings;
    daqp_default_settings(&settings);
    settings.iter_limit = 10000; /*10000 denotes total number of iterations*/
    settings.cycle_tol = 10;   /*1000 denotes allowed number of iterations without progress before terminating*/
    settings.progress_tol = 1e-6; /*1e-12 denotes minimum change in objective function to consider it progress*/
    settings.pivot_tol = 1e-12; /*1e-12 denotes minimum change in objective function to consider it progress*/
    settings.primal_tol = 1.0e-12;
    settings.dual_tol = 1.0e-12;
    settings.rho_soft = 0.0;
    double sol[nVariables];
    double lam[nConstraints + nVariables];
    for (std::size_t i = 0; i < nConstraints + nVariables; ++i) {
        if (i < nVariables) {
            sol[i] = 0.0;
        }
        lam[i] = 0.0;
    }
    std::vector<double> cV = input.m_cV;
    DAQPProblem qp = {nVariables, nConstraints + nVariables, nSimpleConstraints,
                      hessFlatten.data(), cV.data(), jacFlatten.data(),
                      upper.data(), lower.data(), constraintTypes.data()};
    DAQPResult result;
    result.x = sol;   // primal variable
    result.lam = lam; // dual variable
    daqp_quadprog(&result, &qp, &settings);
    output.exitStatus = result.exitflag;
    if (result.exitflag == -2 ||
        result.exitflag == -5 ||
        result.exitflag == -3 ||
        result.exitflag == -6 ||
        result.exitflag == -1) {
        return;
    }
    output.m_cost = result.fval;
    output.m_x = std::vector<double>(result.x, result.x + nVariables);
    output.m_lambda.resize(nConstraints, 0.0);
    output.m_lambdaL.resize(nVariables, 0.0);
    output.m_lambdaU.resize(nVariables, 0.0 );
    for (std::size_t i = nVariables; i < nConstraints + nVariables; ++i) {
        output.m_lambda[i - nVariables] = -result.lam[i]; // lambda for constraints
    }
    const double boundRegion = 1.0e-6;
    for (std::size_t i = 0; i < nVariables; i++) {
        if (std::fabs(upper[i] - output.m_x[i]) < boundRegion) {
            output.m_lambdaU[i] = result.lam[i];
        } else {
            output.m_lambdaL[i] = -result.lam[i];
        }
    }
}
#endif
  
void solveQP(const std::string& logfile, unsigned int iteration,
            QP_SOLVER_TYPE solverType, FSQP_QP_PROBLEM_TYPE fsqpQPType, QPOutput& output) {
    using namespace FSQP_LOG_PARSER;
    IterationData iterData;
    ReadIteration(logfile, iteration, iterData);
    QPInput input;
    input.m_H = iterData.m_hessianD0;
    input.m_A = iterData.m_jacobianD0;
    input.m_cV = iterData.m_cVectorD0;
    input.m_bV = iterData.m_bVectorD0;
    input.m_lower = iterData.m_lower;
    input.m_upper = iterData.m_upper;
    assert(input.m_A.size() == input.m_bV.size());
    assert(input.m_lower.size() == input.m_upper.size());
    const std::size_t nX = input.m_lower.size();
    std::size_t hessSize = 0;
    if (fsqpQPType == FSQP_QP_PROBLEM_TYPE::D0) {
        input.m_lower = iterData.m_lowerD0;
        input.m_upper = iterData.m_upperD0;
        hessSize = nX + 1;
    } else {
        hessSize = nX;
    }
    assert(input.m_H.size() ==  hessSize);
    if  (fsqpQPType == FSQP_QP_PROBLEM_TYPE::D0) {
        input.m_H.resize(nX);
        for (auto& row : input.m_H) {
            assert(row.size() == hessSize);
            row.resize(nX);
        }
    }
    std::size_t nVars = input.m_H.size();
    assert(input.m_cV.size() == nX); 
    for (std::size_t iConstr = 0; iConstr < input.m_A.size(); ++iConstr) {
        assert(input.m_A[iConstr].size() == nX);
    }
    input.m_x0 = std::vector<double>(nX, 0.0);
    if (solverType == QP_SOLVER_TYPE::QLD) {
        solveQLD(input, output);
    } else {
#ifdef DAQP
        solveDAQP(input, output);
#endif
    }

    ASSERT_EQ(iterData.m_d0.size(), output.m_x.size());

    for (std::size_t i = 0; i < output.m_x.size(); ++i) {
        EXPECT_GT(output.m_x[i], input.m_lower[i]) << "iteration/i = " << iteration << "/" << i;
        EXPECT_LT(output.m_x[i], input.m_upper[i]) << "iteration/i = " << iteration << "/" << i;
        EXPECT_NEAR(iterData.m_d0[i], output.m_x[i], 1e-6) << "iteration/i = " << iteration << "/" << i;
    }
    ASSERT_EQ(input.m_A.size(), output.m_lambda.size());
    for (std::size_t i = 0; i < input.m_A.size(); ++i) {
        EXPECT_NEAR(iterData.m_lambdaD0[i], output.m_lambda[i], 1e-4) << "iteration/i = " << iteration << "/" << i;
    }
    ASSERT_EQ(nX, output.m_lambdaU.size());
    ASSERT_EQ(nX, output.m_lambdaL.size());
    for (std::size_t i = 0; i < nX; ++i) {
        EXPECT_NEAR(iterData.m_lambdaD0[input.m_A.size() + i], output.m_lambdaL[i], 1e-4) << "iteration/i = " << iteration << "/" << i;
        EXPECT_NEAR(iterData.m_lambdaD0[input.m_A.size() + nX + i], output.m_lambdaU[i], 1e-4) << "iteration/i = " << iteration << "/" << i;
    }
}
}