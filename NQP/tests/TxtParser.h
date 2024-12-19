#ifndef TXTPARSER_H
#define TXTPARSER_H
#include <iostream>
#include <fstream>
#include "types.h"


namespace TXT_QP_PARSER {

const unsigned int MAX_VALUE_LENGTH = 20U;
const unsigned int MAX_VALUES = 1000000U;
const unsigned int BUFFER_SIZE = 1.5 * MAX_VALUE_LENGTH * MAX_VALUES;

class TxtParser {
public:
    TxtParser();
    ~TxtParser() = default;
    const QP_NNLS::DenseQPProblem& Parse(const std::string& file, bool& status);
private:
    const unsigned int nstages = 5;
    std::string file;
    std::vector<bool> stages;
    QP_NNLS::DenseQPProblem problem;
    std::ifstream fid;
    std::string buf;
    unsigned int fSize = 0;
    unsigned int curPos = 0;
    bool OpenFile(const std::string& file);
    void ParseHessian();
    void ParseJacobian();
    void ParseCVector();
    void ParseLb();
    void ParseUb();
    bool ReadMatrix(QP_NNLS::matrix_t& m);
    bool ReadVector(std::vector<double>& v);
    char FindNextToken();
};
}


#endif // TXTPARSER_H
