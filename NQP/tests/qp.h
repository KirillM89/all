#include <string>
#include <vector>
#include "qp_utils.h"
#ifndef NNLS_TESTS_QP_H
#define NNLS_TESTS_QP_H
using namespace FSQP_LOG_PARSER;
namespace QP_SOLVERS {
    using matrix_t = std::vector<std::vector<double>>;
    enum QP_SOLVER_TYPE {
        QLD = 0,
        DAQP
    };

    struct QPInput {
        std::vector<double> m_x0;
        std::vector<double> m_cV;
        std::vector<double> m_bV;
        std::vector<double> m_lower;
        std::vector<double> m_upper;
        matrix_t m_H; 
        matrix_t m_A;
    };
    struct QPOutput {
        std::string m_errMsg;
        double m_cost;
        int exitStatus; // 0 - OK, 1 - INFEASIBLE
        std::vector<double> m_x;
        std::vector<double> m_lambda;
        std::vector<double> m_lambdaU;
        std::vector<double> m_lambdaL;
    };
    //bool checkInput(const FSQP_LOG_PARSER::IterationData& input); 
    std::vector<double> FlattenByColumns(const matrix_t& matrix);
    void solveQP(const std::string& logfile, unsigned int iteration,
                 QP_SOLVER_TYPE solverType, FSQP_QP_PROBLEM_TYPE fsqpQPType,
                 QPOutput& output);
    void solveQLD(const QPInput& input, QPOutput& output);
}

#endif