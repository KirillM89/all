#include "TxtParser.h"
#include <cassert>
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

} //namespace TXT_QP_PARSER
