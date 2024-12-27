
#include "qp_utils.h"
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>
namespace  FSQP_LOG_PARSER {
    std::vector<std::string> split(const std::string& str, char delimeter)
    {
        std::size_t curPos = 0;
        std::vector<std::string> output;
        while (curPos < str.size()) {
            std::size_t newPos = str.find(delimeter, curPos);
            if (newPos > curPos) {
                if (newPos == std::string::npos) {
                    newPos = str.size();
                }
                output.push_back(str.substr(curPos, newPos - curPos));
            }
            curPos = newPos + 1;
        }
        return output;
    }

    bool findVector(std::size_t pos, const std::string& str, std::size_t& bg, std::size_t& end) 
    {
        const char vectorBegin = '[';
        const char vectorEnd = ']';
        const std::size_t pBg = str.find(vectorBegin, pos);
        if (pBg == std::string::npos) {
            return false;
        }
        const std::size_t pEnd = str.find(vectorEnd, pos);
        if (pEnd == std::string::npos || pEnd <= pBg) {
            return false;
        }
        bg = pBg;
        end = pEnd;
        return true;
    }
    bool readVector(std::size_t pos, const std::string& str, std::vector<double>& output) 
    {
        output.clear();
        std::size_t pBg, pEnd;
        if (!findVector(pos, str, pBg, pEnd)) {
            return false;
        } 
        const char delimeter =',';
        std::string num;
        std::size_t curPos = pBg + 1;
        while (curPos <= pEnd) {
            std::size_t delPos = str.find(delimeter, curPos);
            if (delPos == std::string::npos || delPos > pEnd) {
                delPos = pEnd;
            }
            if (curPos >= delPos - 1){
                return false;
            }
            num = str.substr(curPos, delPos - curPos);
            double val;
            try{
                val = std::stod(num);
            } catch (...) {
                return false;
            }    
            output.push_back(val);
            curPos = delPos + 1;
        }
        return true;
    }
    bool removeExcept(std::string& str)
    { 
        for (std::string::iterator it = str.begin(); it != str.end();){
            if (*it < '0' || *it > '9') {
                str.erase(it);
            }
            else {
                ++it;
            }
        }
        return true;
    }
    int getMatrixRowIndex(const std::string& str, std::size_t pos, std::size_t& pVcBg)
    {
        const char rowBg = ']';
        std::size_t curPos = pos;
        std::size_t rowBgPos = str.find(rowBg, curPos);
        if (rowBgPos == std::string::npos){
            return -1;
        }
        std::string prefix = str.substr(curPos, rowBgPos - curPos);
        std::vector<std::string> splt = split(prefix);
        if (splt.empty()) {
            return -1;
        }
        int index = std::stoi(splt.back());
        pVcBg = rowBgPos + 1;
        return index;

    }
    bool readMatrixRow(std::size_t pBg, const std::string& str, std::vector<double>& output, std::size_t& pEnd, int& rowIndex)
    { 
        std::size_t pVcBg = 0;
        const char columnNum = ')';
        const char newLine = '\n';
        rowIndex = getMatrixRowIndex(str, pBg, pVcBg);
        if (rowIndex == -1){
            return false;
        }
        std::size_t pVcEnd = str.find(newLine, pVcBg);
        if (pVcEnd == std::string::npos) {
            return false;
        }
        std::string row = str.substr(pVcBg, pVcEnd - pVcBg);
        std::vector<std::string> items = split(row);
        if (items.size() % 2 != 0) {
            return false;
        }
        output.clear();
        for (std::size_t i = 0; i < items.size() / 2; ++i) {
            if (items[2 * i].back() != columnNum) {
                return false;
            }
            int columnIndex = std::stoi(items[2 * i].substr(0, items[2 * i].size() - 1));
            while (columnIndex > output.size()) {
                output.push_back(0.0);
            }
            output.push_back(std::stod(items[2 * i + 1]));
        }
        pEnd = pVcEnd;
        return true;
    }

    bool readMatrix(std::size_t pos, const std::string& str, int nRows, matrix_t& output, std::size_t& posEnd)
    {
        output.resize(nRows);
        std::size_t pBg = pos;
        std::size_t pEnd = std::string::npos;
        int rowIndex = -1;
        std::size_t maxRowLen = std::numeric_limits<double>::min();
        for (std::size_t iRow = 0; iRow < nRows; ++iRow) {
            if (!readMatrixRow(pBg, str, output[iRow], pEnd, rowIndex)) {
                return false;
            }
            if (iRow != rowIndex) {
                return false;
            }
            maxRowLen = std::max(maxRowLen, output[iRow].size());      
            pBg = pEnd + 1;       
        }
        for (std::size_t iRow = 0; iRow < nRows; ++iRow) {
            if (output[iRow].size() < maxRowLen) {
                output[iRow].resize(maxRowLen, 0.0);
            }
        } 
        posEnd = pBg;
        return true;
    }
    std::size_t getIterationBlockPosition(unsigned int iteration, const std::string& buffer)
    {
        const std::string iter = "ITERATION:";
        std::size_t pos = 0;
        while(pos < buffer.size()) {
            pos = buffer.find(iter, pos);
            if (pos == std::string::npos) {
                return 0;
            }
            pos += iter.size();
            std::size_t nlPos = buffer.find('\n', pos);
            std::string iterIndex = buffer.substr(pos, nlPos - pos);
            std::vector<std::string> items = split(iterIndex);
            if (items.size() != 1) {
                return 0;
            }
            if (iteration == std::stoi(items[0])){
                return nlPos + 1;
            }
            pos = nlPos;
        }
    }
    std::size_t getPosition(const std::string& buffer, const std::string& pattern, std::size_t pBg, std::size_t pEnd) 
    {
        std::string substr = buffer.substr(pBg, pEnd - pBg);
        std::size_t pos = substr.find(pattern, 0);
        if (pos == std::string::npos) {
            return 0;
        }
        pos += pattern.size();
        if (pos > substr.size()) {
            return 0;
        }
        return pos + pBg;
    }
    
    std::string readIntoString(const std::string& fileName) {
        std::ifstream fid(fileName, std::ifstream::binary);
        fid.clear();
        fid.seekg(0, std::ios::end);
        std::size_t size = fid.tellg();
        fid.seekg(0, std::ios::beg);
        if (size == 0) {
            return std::string();
        }
        std::string buffer(size, ' ');
        fid.read(&buffer[0], size);
        fid.close();
        return buffer;
    }
    bool ReadIteration(const std::string& fileName, unsigned int iteration, IterationData& input)
    {   
        const std::string shiftFromPrevLine = "\n  ";
        const std::string sD0 = shiftFromPrevLine + "d0:";
        const std::string sLowerBounds = shiftFromPrevLine + "lowerBounds:";
        const std::string sUpperBounds = shiftFromPrevLine + "upperBounds:";
        const std::string sCVectorD0 = shiftFromPrevLine + "m_cVectorForD0:";
        const std::string sBVectorD0 = shiftFromPrevLine + "m_bVectorForD0:";
        const std::string sJacobianD0 = shiftFromPrevLine + "jacobian for d0:";
        const std::string sLambdaD0 = shiftFromPrevLine + "lambda d0:";
        const std::string sHessian = shiftFromPrevLine + "hessian:";
        const std::string sLowerBoundsD0 = shiftFromPrevLine + "lower bounds d0:";
        const std::string sLowerBoundsDTil = shiftFromPrevLine + "lower bounds dTil:";
        const std::string sUpperBoundsD0 = shiftFromPrevLine + "upper bounds d0:";
        const std::string sUpperBoundsDTil = shiftFromPrevLine + "upper bounds dTil:";
        const std::string sX = shiftFromPrevLine + "x:";

        const std::string buffer = readIntoString(fileName);
        std::size_t lbPos = buffer.find(sLowerBounds);
        if (lbPos == std::string::npos) {
            return false;
        }
        lbPos = lbPos + sLowerBounds.size();
        std::size_t ubPos = buffer.find(sUpperBounds);
        if (ubPos == std::string::npos) {
            return false;
        }
        ubPos = ubPos + sUpperBounds.size();
        readVector(lbPos, buffer, input.m_lower);
        readVector(ubPos, buffer, input.m_upper);
        std::size_t pBg = getIterationBlockPosition(iteration, buffer);
        if (pBg == 0) {
            return false;
        }
        std::size_t pEnd = getIterationBlockPosition(iteration + 1, buffer);
        if (pEnd == 0) {
            pEnd = buffer.size();
        }
        std::size_t pos = getPosition(buffer, sX, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readVector(pos, buffer, input.m_x);
        const std::size_t nX = input.m_x.size();
        if (nX == 0) {
            return false;
        }
        pos = getPosition(buffer, sD0, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readVector(pos, buffer, input.m_d0);

        pos = getPosition(buffer, sCVectorD0, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readVector(pos, buffer, input.m_cVectorD0);

        pos = getPosition(buffer, sBVectorD0, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readVector(pos, buffer, input.m_bVectorD0);

        pos = getPosition(buffer, sLambdaD0, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readVector(pos, buffer, input.m_lambdaD0);

        pos = getPosition(buffer, sLowerBoundsD0, pBg, pEnd);
        if (pos != 0) {
            readVector(pos, buffer, input.m_lowerD0);
        }

        pos = getPosition(buffer, sLowerBoundsDTil, pBg, pEnd);
        if (pos != 0) {
            readVector(pos, buffer, input.m_lowerDTil);
        }

        pos = getPosition(buffer, sUpperBoundsD0, pBg, pEnd);
        if (pos != 0) {
            readVector(pos, buffer, input.m_upperD0);
        }

        pos = getPosition(buffer, sUpperBoundsDTil, pBg, pEnd);
        if (pos != 0) {
            readVector(pos, buffer, input.m_upperDTil);
        }

        const std::size_t nLambda = input.m_lambdaD0.size();
        std::size_t nConstraints = nLambda - 2 * nX;
        std::size_t posEnd = 0;

        pos = getPosition(buffer, sJacobianD0, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        readMatrix(pos, buffer, nConstraints, input.m_jacobianD0, posEnd);
        pos = getPosition(buffer, sHessian, pBg, pEnd);
        if (pos == 0) {
            return false;
        }
        const std::size_t hessianSize = nX + 1;
        readMatrix(pos, buffer, hessianSize, input.m_hessianD0, posEnd);
        
        return true;
    }



    bool QPOutputReader(const std::string& fileName, unsigned int iteration, FSQP_QP_PROBLEM_TYPE type, QPProblemOutput& output)
    {
        return true;
    }
}