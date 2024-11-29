#ifndef  NNLS_TESTS_QP_UTILS_H
#define  NNLS_TESTS_QP_UTILS_H
#include <vector>
#include <string>

namespace FSQP_LOG_PARSER {
using matrix_t = std::vector<std::vector<double>>;
struct IterationData {
    matrix_t m_hessianD0;
    matrix_t m_jacobianD0;
    matrix_t m_jacobianD1;
    std::vector<double> m_cVectorD0;
    std::vector<double> m_cVectorD1;
    std::vector<double> m_cVectorDtil;
    std::vector<double> m_bVectorD0;
    std::vector<double> m_d0;
    std::vector<double> m_d1;
    std::vector<double> m_dTil;
    std::vector<double> m_lambdaD0;
    std::vector<double> m_lambdaD1;
    std::vector<double> m_lambdaDTil;
    std::vector<double> m_x;
    std::vector<double> m_lower;
    std::vector<double> m_upper;
    std::vector<double> m_lowerD0;
    std::vector<double> m_lowerDTil;
    std::vector<double> m_upperD0;
    std::vector<double> m_upperDTil;
};
struct QPProblemOutput {
    std::vector<double> m_x;
    std::vector<double> m_lambdaU;
    std::vector<double> m_lambdaL;
    std::vector<double> m_lambdaC;
    double cost;
};
enum class FSQP_QP_PROBLEM_TYPE {
    D0 = 0,
    D1,
    DTil
};
std::vector<std::string> split(const std::string& str, char delimenter = ' ');
std::size_t getPosition(const std::string& buffer, const std::string& pattern, std::size_t pBg, std::size_t pEnd);
std::size_t getIterationBlockPosition(unsigned int iteration, const std::string& fid);
std::string readIntoString(const std::string& fileName);
bool removeExcept(std::string& str);
int getMatrixRowIndex(const std::string& str, std::size_t pos, std::size_t& pVcBg);
bool findVector(std::size_t pos, const std::string& str, std::size_t& bg, std::size_t& end);
bool readVector(std::size_t pos, const std::string& str, std::vector<double>& output);
bool readMatrixRow(std::size_t pos, const std::string& str, std::vector<double>& output, std::size_t& pEnd, int& rowIndex);
bool readMatrix(std::size_t pos, const std::string& str, int nRows, matrix_t& output, std::size_t& posEnd);
bool ReadIteration(const std::string& logfileName, unsigned int iteration, IterationData& input);
bool QPOutputReader(const std::string& fileName, unsigned int iteration, FSQP_QP_PROBLEM_TYPE type, QPProblemOutput& output);
}
#endif