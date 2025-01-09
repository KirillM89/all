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
template <> struct IsDumpAble<unsigned int&> {
    static const bool value = true;
};
template <> struct IsDumpAble<const char*> {
    static const bool value = true;
};
template <> struct IsDumpAble<const std::string&> {
    static const bool value = true;
};
template <> struct IsDumpAble<std::string&> {
    static const bool value = true;
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
        if ((--deep) == 0) {
            Dump(std::string_view(fmt), std::make_format_args(sym, args...));
            fmt.clear();
        }
    }
    template<typename T, std::enable_if_t<IsDumpAble<T>::value, bool> = true> constexpr void Write(T&& sym) {
        if (std::is_same_v<T, double&>) {
            Append(fmtDouble);
        } else if (std::is_same_v<T, int&> || std::is_same_v<T, unsigned int&> ) {
            Append(fmtInt);
        } else if (std::is_same_v<T, const std::string&> ||
                   std::is_same_v<T, const char*> ||
                   std::is_same_v<T, std::string&>) {
            Append(fmtStr);
        }
        if ((deep) == 0) {
            Dump(std::string_view(fmt), std::make_format_args(sym));
            fmt.clear();
        }
    }
    void Write(const char* sym) {
        Append(fmtStr);
        if ((deep) == 0) {
            Dump(std::string_view(fmt), std::make_format_args(sym));
            fmt.clear();
        }
    }
    void Write(const std::string& sym) {
        Append(fmtStr);
        if ((deep) == 0) {
            Dump(std::string_view(fmt), std::make_format_args(sym));
            fmt.clear();
        }
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
    void Dump(std::string_view fmt, std::format_args&& args) {
        fid << std::vformat(fmt, args);
    }
};
}

#endif // DATA_WRITER_H
