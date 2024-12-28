#ifndef DATA_WRITER_H
#define DATA_WRITER_H
#include <vector>
#include <format>
#include <string_view>
#include <iostream>
#include <fstream>
namespace FMT_WRITER {
class FmtWriter {

public:
    FmtWriter() = default;
    ~FmtWriter() = default;
    void SetFile(const std::string& fileName);
    template<typename T, typename ...Args> constexpr void Write(T&& sym, Args&&... args) {
        if (std::is_same_v<T, std::string>) {
            Append(fmtStr);
        } else if (std::is_same_v<T, double>) {
            Append(fmtDouble);
        } else if (std::is_same_v<T, int>) {
            Append(fmtInt);
        }
        Write(args...);
        Dump(std::string_view(fmt), std::make_format_args(sym, args...));
    }
    template<typename T> constexpr void Write(T&& sym) {
        if (std::is_same_v<T, std::string>) {
            Append(fmtStr);
        } else if (std::is_same_v<T, double>) {
            Append(fmtDouble);
        } else if (std::is_same_v<T, int>) {
            Append(fmtInt);
        }

    }
    void Reset();
private:
    std::size_t nHeaders;
    std::ofstream fid;
    std::string fmt;
    const std::string fmtDouble = "{:.10f}";
    const std::string fmtInt = "{:10}";
    const std::string fmtStr = "{:10}";
    constexpr void Append(const std::string& tail) {
        fmt += " " + tail;
    }
    void Dump(std::string_view fmt, std::format_args&& args) {
        fid << std::vformat(fmt, args);
    }
};
}

#endif // DATA_WRITER_H
