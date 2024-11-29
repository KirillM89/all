#ifndef NNLS_QP_SOLVER_LOG_H
#define NNLS_QP_SOLVER_LOG_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include "types.h"
#define CPP_FORMAT
namespace QP_NNLS {
#define SEP " "
	template<typename T> std::ostream& operator << (std::ostream& f, const std::vector<T>& v) {
		#ifdef CPP_FORMAT
		f << std::setprecision(15);
		f << "{" << SEP;
		#endif
		for (const auto& el : v) {
			#ifndef CPP_FORMAT
			f << SEP << el;
			#else
			if (&el != &v.back()) {
				f << el << ", ";
			} else {
				f << el;
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
}
#endif