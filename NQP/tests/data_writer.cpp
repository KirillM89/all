#include <cassert>
#include "data_writer.h"
namespace FMT_WRITER {

void FmtWriter::SetFile(const std::string& fileName) {
    fid.open(fileName, std::ios::out);
    assert(fid.is_open());
}

void FmtWriter::Reset() {
    nHeaders = 0;
    fmt.clear();
}

} //namespace FMT_WRITER
