#include "TxtParser.h"
#include <cassert>
#include <cmath>
namespace TXT_QP_PARSER {
TxtParser::TxtParser():
    file(),
    stages(std::vector<bool>(nstages, false))
{}
const QP_NNLS::DenseQPProblem& TxtParser::Parse(const std::string& file, bool& status) {
    stages = std::vector<bool>(nstages, false);
    if (OpenFile(file)) {
        status = true;
        try {
            ParseHessian();
            ParseJacobian();
            ParseCVector();
            ParseLb();
            ParseUb();
            for (auto res: stages) {
                if (!res) {
                    status = false;
                }
            }
        } catch (...) {
            fid.close();
            status = false;
        }
        fid.close();
    } else {
        status = false;
    }
    return problem;
}

bool TxtParser::OpenFile(const std::string& file) {
    fid = std::ifstream(file , std::ifstream::in);
    if (fid.is_open()) {
        fid.seekg(0, fid.end);
        fSize = fid.tellg();
        curPos = 0;
        fid.seekg(0, fid.beg);
        return true;
    }
    return false;
}

void TxtParser::ParseHessian() {
    if (ReadMatrix(problem.H)) {
        stages[0] = true;
    }
}
void TxtParser::ParseJacobian() {
    if (stages[0] && ReadMatrix(problem.A)) {
        stages[1] = true;
    }
}
void TxtParser::ParseCVector() {
    if (stages[1] && ReadVector(problem.c)) {
        stages[2] = true;
    }
}
void TxtParser::ParseLb() {
    if (stages[2] && ReadVector(problem.lw)) {
        stages[3] = true;
    }
}
void TxtParser::ParseUb(){
    if (stages[3] && ReadVector(problem.up)) {
        stages[4] = true;
    }
}
bool TxtParser::ReadMatrix(QP_NNLS::matrix_t& m) {
    m.clear();
    assert(fSize > curPos);
    const unsigned int rSize = std::min(BUFFER_SIZE, fSize - curPos);
    buf.resize(rSize);
    fid.read(&buf[0], rSize);
    bool matBgFound = false;
    while (true) {
        const char nextToken = FindNextToken();
        if (!matBgFound) {
            if (nextToken == '{') {
                matBgFound = true;
            }
        } else {
            if (nextToken == '{') {
                m.emplace_back();
                ReadVector(m.back());
                if (m.size() > 1 && (m.back().size() != m[m.size() - 2].size())) {
                    return false;
                }
            } else if (nextToken == '}') {
                ++curPos;
                return true;
            } else {
                abort();
            }
        }
    }
    return false;
}
bool TxtParser::ReadVector(std::vector<double>& v) {
    v.clear();
    bool bgFound = false;
    std::string value;
    while (curPos < fSize) {
        const char el = buf[curPos];
        if (!bgFound) {
            if (el == '{') {
                bgFound = true;
            }
        } else {
            if (el == ',') {
                if (!value.empty()) {
                    v.push_back(std::stod(value));
                }
                value.clear();
            }
            else if (('0' <= el && el <= '9') || el == '.' || el == '-' || el == 'e' || el == '+'){
                value.push_back(el);
            }
            else if (el == '}') {
                if (!value.empty()) {
                    v.push_back(std::stod(value));
                }
                ++curPos;
                return true;
            }
        }
        ++curPos;
    }
    return false;
}
char TxtParser::FindNextToken() {
    while (curPos < fSize) {
        const char token = buf[curPos];
        if (token == '{' || token == '}') {
            return token;
        }
        ++curPos;
    }
    return '!'; // not found
}

DenseProblemFormatter::DenseProblemFormatter() = default;

const QP_NNLS::DenseQPProblem& DenseProblemFormatter::PrepareProblem(const std::string& fileName,
                                                                     DENSE_PROBLEM_FORMAT fmt) {
    using namespace QP_NNLS;
    output.status= false;
    pt = txtParser.Parse(fileName, output.status);
    if (!output.status) {
        output.errMsg = "Failed to read problem: " + fileName;
        return problem;
    }
    const std::size_t nV = pt.H.size();
    for (const auto& row: pt.H) {
        if (row.size() != nV) {
            output.errMsg = "hessian is not square matrix";
            return problem;
        }
    }
    const std::size_t nC = pt.A.size();
    for (const auto& row: pt.A) {
        if (row.size() != nV) {
            output.errMsg = "length of Jacobian row n.e. to number of variables";
            return problem;
        }
    }
    if (pt.c.size() != nV) {
        output.errMsg = "size of c vector n.e. to n variables";
        return problem;
    }
    if (pt.lw.size() != pt.up.size()) {
        output.errMsg = "number of lower bounds n.e. to number of upper bounds";
        return problem;
    }
    const std::size_t nB = pt.lw.size();
    if (nB < nV) {
        output.errMsg = "number of bounds l.t. number of variables";
        return problem;
    }
    std::size_t nEq = 0;
    for (std::size_t i = 0; i < nB; ++i) {
        if (pt.lw[i] > pt.up[i]) {
            output.errMsg = "lower bound g.t. upper bound";
            return problem;
        } else if (std::fabs(pt.lw[i] - pt.up[i]) < 1.0e-8) {
            ++nEq;
        }
    }
    assert(nB == nC && nC >= nV);
    problem.nEqConstraints = nEq;
    problem.H = pt.H;
    problem.c = pt.c;
    GenJac(fmt);
    GenBnds();
    output.status = true;
    output.errMsg.clear();
    return problem;
}

void DenseProblemFormatter::GenJac(DENSE_PROBLEM_FORMAT fmt) {
    const std::size_t nv = pt.H.size();
    const std::size_t nc = pt.A.size();
    const std::size_t nb = pt.lw.size();
    const std::size_t nConstraints = nb - nv;
    if (fmt == DENSE_PROBLEM_FORMAT::RIGHT) {
        problem.A.resize(2 * nConstraints, std::vector<double>(nv));
        problem.b.resize(2 * nConstraints);
        for (std::size_t i = 0; i < nConstraints; ++i) {
            for (std::size_t j = 0; j < nv; ++j) {
                problem.A[2 * i][j] = pt.A[i][j];
                problem.A[2 * i + 1][j] = -pt.A[i][j];
                problem.b[2 * i] = pt.up[i];
                problem.b[2 * i + 1] = -pt.lw[i];
            }
        }
    } else if (fmt == DENSE_PROBLEM_FORMAT::LEFT_RIGHT) {
        problem.A = QP_NNLS::matrix_t(pt.A.begin(), pt.A.begin() + nConstraints);
    }
}

void DenseProblemFormatter::GenBnds() {
    const std::size_t nv = pt.H.size();
    const std::size_t nc = pt.A.size();
    problem.lw.resize(nv);
    problem.up.resize(nv);
    std::size_t ii = 0;
    for (std::size_t i = nc - nv; i < nc; ++i) {
        problem.lw[ii] = pt.lw[i];
        problem.up[ii] = pt.up[i];
        ++ii;
    }
}


} //namespace TXT_QP_PARSER
