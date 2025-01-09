#ifndef DATA_WRITER_H
#define DATA_WRITER_H
#include <vector>
#include <format>
#include <string_view>
#include <iostream>
#include <fstream>
#include <cassert>
namespace FMT_WRITER {

template <typename T> struct IsDumpAble {
    static const bool value = false;
};

template <> struct IsDumpAble<double&> {
    static const bool value = true;
};
template <> struct IsDumpAble<int&> {
    static const bool value = true;
};

class FmtWriter {

public:
    FmtWriter() = default;
    ~FmtWriter() = default;
    void SetFile(const std::string& fileName);
    template<typename T, typename ...Args> constexpr void Write(T&& sym, Args&&... args) {
        Write(sym);
        ++deep;
        Write(args...);
        if ((--deep) == 0) {
            Dump(std::string_view(fmt), std::make_format_args(sym, args...));
            fmt.clear();
        }
    }
    template<typename T, std::enable_if_t<IsDumpAble<T>::value, bool> = true> constexpr void Write(T&& sym) {
        if (std::is_same_v<T, double&>) {
            Append(fmtDouble);
        } else if (std::is_same_v<T, int&>) {
            Append(fmtInt);
        }
    }
    void Write(const char* sym) {
        Append(fmtStr);
    }
    void Reset();
private:
    unsigned int deep = 0;
    std::size_t nHeaders;
    std::ofstream fid;
    std::string fmt;
    const std::string fmtDouble = "{:>20.5e}";
    const std::string fmtInt = "{:>20}";
    const std::string fmtStr = "{:>20s}";
    constexpr void Append(const std::string& tail) {
        fmt += " " + tail;
    }
    void Dump(std::string_view fmt, std::format_args&& args) {
        fid << std::vformat(fmt, args) << std::endl;
    }
};
}

#endif // DATA_WRITER_H
