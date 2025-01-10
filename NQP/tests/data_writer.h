#ifndef DATA_WRITER_H
#define DATA_WRITER_H
#include <vector>
#include <format>
#include <string_view>
#include <iostream>
#include <fstream>
#include <cassert>
namespace FMT_WRITER {

template <typename T> struct IsStringType {
    static const bool value = std::is_convertible_v<T, const std::string&>;
};

template<typename T> struct IsFloatType {
    static const bool value = false;
};
template <> struct IsFloatType<double&> {
    static const bool value = true;
};
template <> struct IsFloatType<float&> {
    static const bool value = true;
};
template <typename T> struct IsIntegralType {
    static const bool value = false;
};
template <> struct IsIntegralType<int&> {
    static const bool value = true;
};
template <> struct IsIntegralType<unsigned int&> {
    static const bool value = true;
};
template <typename T> struct IsDumpAble {
    static const bool value = IsStringType<T>::value ||
            IsIntegralType<T>::value ||
            IsFloatType<T>::value;
};


class FmtWriter {

public:
    FmtWriter() = default;
    ~FmtWriter() {
        fid.close();
    };
    void SetFile(const std::string& fileName);
    template<typename T, typename ...Args> constexpr void Write(T&& sym, Args&&... args) {
        ++deep;
        Write(sym);
        Write(args...);
        --deep;
        ToFile(sym, args...);
    }
    template<typename T, std::enable_if_t<IsDumpAble<T>::value, bool> = true> constexpr void Write(T&& sym) {
        if (IsFloatType<T>::value) {
            Append(fmtDouble);
        } else if (IsIntegralType<T>::value) {
            Append(fmtInt);
        } else if (IsStringType<T>::value) {
            Append(fmtStr);
        }
        ToFile(sym);
    }

    void Reset();
    void NewLine() {
        fid << std::endl;
        ++currentLine;
    }
    unsigned int LineNumber() {
        return currentLine;
    }
private:
    unsigned int deep = 0;
    static unsigned int currentLine;
    std::ofstream fid;
    std::string fmt;
    const std::string fmtDouble = "{:>20.5e}";
    const std::string fmtInt = "{:>20}";
    const std::string fmtStr = "{:>20s}";
    constexpr void Append(const std::string& tail) {
        fmt += " " + tail;
    }
    template<typename ...Args> void ToFile(Args&&... args) {
        if (deep == 0) {
            Dump(std::string_view(fmt), std::make_format_args(args...));
            fmt.clear();
        }
    }
    void Dump(std::string_view fmt, std::format_args&& args) {
        fid << std::vformat(fmt, args);
    }

};
}

#endif // DATA_WRITER_H
