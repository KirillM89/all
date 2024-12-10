#ifndef NNLS_QP_SOLVER_LOG_H
#define NNLS_QP_SOLVER_LOG_H
#include <iostream>
#include <fstream>
#include <type_traits>
#include <iomanip>
#include "types.h"
#include "decorators.h"
#define CPP_FORMAT
namespace QP_NNLS {
#define SEP " "
template<class T, template <class ...> class C> struct IsLogAble {
    static const bool value = false;
};
template <class T> struct IsLogAble <T, std::vector> {
    static const bool value = true;
};
template <class T> struct IsLogAble <T, std::unordered_set> {
    static const bool value = true;
};
template<class T, template <class ...> class M,
std::enable_if_t<IsLogAble<T, M>::value, bool> = true> std::ostream& operator << (std::ostream& f, const M<T>& v) {
    #ifdef CPP_FORMAT
    f << std::setprecision(15);
    f << "{";
    #endif
    typename M<T>::const_iterator it;
    const std::size_t sz = v.size();
    std::size_t counter = 0;
    for (it = v.begin(); it != v.end(); ++it) {
        #ifndef CPP_FORMAT
        f << SEP << *it << " ";
        #else
        if (++counter < sz) {
            f << *it << ", ";
        } else {
            f << *it;
        }
        #endif
    }
    #ifndef CPP_FORMAT
    f << "\n";
    #else
    f << "}" << "\n";
    #endif
    return f;
}

std::ostream& operator << (std::ostream& f, const matrix_t& mat);

class Logger
{
public:
    Logger() = default;
    virtual ~Logger() {
        fid << std::endl;
        fid.close();
    }
    bool SetFile(const std::string& filename);
    void SetStage(const std::string& stageName);
    void PrintActiveSetIndices(const std::vector<int>& indices);
    void CloseFile() {
        fid.close();
    }
    template<typename T> void dump(const std::string& description, const T& obj) {
        fid << description << "\n";
        fid << obj;
    }
    template<typename T, typename ...Args> void message(T arg, Args ...args) {
        fid << SEP << arg;
        message(args...);
    }
    template<typename T> void message(T arg) {
        fid << SEP << arg << "\n";
    }
private:
    std::ofstream fid;
    std::string logFile;
    void ToNewLine();
};

class Callback1 : public Callback {
public:
    Callback1(const std::string& filePath);
    virtual ~Callback1() override = default;
    void ProcessData(int stage) override;
private:
    std::unique_ptr<Logger> logger;
};

}
#endif
