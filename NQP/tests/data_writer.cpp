#include <cassert>
#include "data_writer.h"
namespace FMT_WRITER {
unsigned int FmtWriter::currentLine = 0;
void FmtWriter::SetFile(const std::string& fileName) {
    if (fid.is_open()) {
        fid.close();
    }
    fid.open(fileName, currentLine == 0 ? std::ios::out : std::ios::app);
    assert(fid.is_open());
}

void FmtWriter::Reset() {
    fmt.clear();
    currentLine = 0;
}

} //namespace FMT_WRITER
