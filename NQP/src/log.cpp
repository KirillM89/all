#include "log.h"
namespace QP_NNLS {

	std::ostream& operator << (std::ostream& f, const matrix_t& mat) {
		#ifdef CPP_FORMAT 
		f << "{";
		#endif
		for (const auto& row : mat) {
			f << row;
			#ifdef CPP_FORMAT
	        if (&row != &mat.back()) {
				f << ",";
			}
			#endif
		}
		#ifdef CPP_FORMAT 
		f << "}\n";
		#endif

		return f;
	}

    bool Logger::SetFile(const std::string& fileName, bool clear) {
        if (fid.is_open()) {
            fid.close();
        }
		logFile = fileName;
        fid.open(logFile, clear ? std::ios::out : std::ios::app);
		return fid.is_open();
	}

	void Logger::SetStage(const std::string& stageName) {
		const char* stageSep = "--------------";
		fid << "\n" << stageSep << stageName << stageSep << "\n";
	}
	void Logger::ToNewLine() {
		fid << "\n";
	}
	void Logger::PrintActiveSetIndices(const std::vector<int>& indices) {
		const int n = static_cast<int>(indices.size());
		fid << "active set indices: ";
		for (int indx = 0; indx < n; ++indx) {
			if (indices[indx] == 1) {
				fid << SEP << indx;
			}
		}
		fid << "\n";
	}
}
