#include "utils.h"
#include <cmath>
#include <iostream>
#ifdef EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

//#define TEST_MODE
namespace QP_NNLS {
	matrix_t& operator-(matrix_t& M) {
		const std::size_t n = M.size();
		if (n == 0) {
			return M;
		}
		const std::size_t m = M.front().size();
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				M[i][j] = -M[i][j];
			}
		}
		return M;
	}
	void ComputeCholFactor(const matrix_t& M, matrix_t& cholF) {
		// A=LLT
		// cholF must be initialized with zeros
		// Cholesky–Banachiewicz 
		std::size_t n = M.size();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				double sum = 0;
				for (int k = 0; k < j; k++) {
					sum += cholF[i][k] * cholF[j][k];
				}
				if (i == j)
					cholF[i][j] = sqrt(M[i][i] - sum);
				else
					cholF[i][j] = (1.0 / cholF[j][j]) * (M[i][j] - sum);
			}
		}
	}
	bool ComputeCholFactorT(const matrix_t& M, matrix_t& cholF, CholetskyOutput& output) {
		// A=L_T * L
		// cholF must be initialized with zeros
		output.negativeBlocking = 1.0; 
		output.negativeDiag.clear();
		output.pivoting = false;
		const std::size_t n = M.size();
		for (int row = n - 1; row >= 0; --row) {
			for (int col = row; col >= 0; --col) {
				double sum = 0.0;
				for (int k = n - 1; k >= row; --k) {
					sum += cholF[k][col] * cholF[k][row];
				}
				double factor = M[row][col] - sum;
				if (col == row) {
					if (std::fabs(factor) < CONSTANTS::cholFactorZero) {
						output.negativeDiag.emplace_back(row, factor);
						factor = CONSTANTS::cholFactorZero;
                        cholF[row][col] = factor; //sqrt(factor);
					} else if (factor < 0.0) {
						output.negativeBlocking = factor;
						return false;
                    } else {
                        cholF[row][col] = sqrt(factor);
                    }
				}
				else {
					cholF[row][col] = (1.0 / cholF[row][row]) * factor;
				}
			}
		}
		return true;
	}
	int ComputeCholFactorTFullPivoting(matrix_t& M, matrix_t& cholF, std::vector<int>& permut) {
		// Computes permutation matrix and modifies M: M = P_T * M * P
		// M = L_T * L 
		// P_T * M * P = P_T * (L_T * L) * P = (L * P)_T * (L * P) = K_T * K
		const std::size_t n = M.size();
		std::size_t nPrmtElements = 0;  // actual number of elements in permutation matrix
		permut.resize(n, -1); // maximum number of elements in permutation matrix is n
		int iRowNew = -1;
		for (int row = n - 1; row >= 0; --row) {
			for (int col = row; col >= 0; --col) {
				double sum = 0.0;
				for (int k = n - 1; k >= row; --k) {
					sum += cholF[k][col] * cholF[k][row];
				}
				double factor = M[row][col] - sum;
#ifdef TEST_MODE
                std::cout << "factor/r/c " << factor << " " << row << " " << col << std::endl;
#endif
				if (col == row) { // first column step
#ifndef TEST_MODE
					if (factor < CONSTANTS::cholFactorZero) { // negative or zero 
#endif
					    // find biggest diagonal element from the unprocessed part of M 
						// if several biggest get closest one
		                double maxMdiag = M[row][col];
						iRowNew = row;
                        for (int l = row - 1; l >= 0; --l) {
                            if (M[l][l] > maxMdiag) {
								iRowNew = l;
								maxMdiag = M[l][l];
							}
						}
						// if index of biggest diagonal element > row 
						// swap M rows and columns, and only rows for CholF P_T * M * P = (L * P)_T * ( L * P)
						if (iRowNew < row) {
							std::swap(M[row], M[iRowNew]);
							swapColumns(M, row, iRowNew);
							swapColumns(cholF, row, iRowNew);
							permut[row] = iRowNew;
							++nPrmtElements;
							//recompute factor
							double sum = 0.0;
							for (int k = n - 1; k >= row; --k) {
								sum += cholF[k][col] * cholF[k][row];
							}
							factor = M[row][col] - sum;
#ifndef TEST_MODE
							if (factor < CONSTANTS::cholFactorZero) {
								return n - row;
							}
#endif
						}
#ifndef TEST_MODE
						else {
							return n - row;
						}
#endif
#ifndef TEST_MODE
					}
#endif
					cholF[row][col] = sqrt(factor);
				}
				else {
					cholF[row][col] = (1.0 / cholF[row][row]) * (factor);
				}
			}
		}
		return 0;
	}

	void Mult(const matrix_t& M1, const matrix_t& M2, matrix_t& mult) { //M1*M2
		const std::size_t n1 = M1.size();
		const std::size_t m1 = M1.front().size();
		const std::size_t m2 = M2.front().size();
		for (std::size_t k = 0; k < n1; ++k) {
			for (std::size_t i = 0; i < m2; ++i) {
				mult[k][i] = 0.0;
				for (std::size_t j = 0; j < m1; ++j) {
					mult[k][i] += M1[k][j] * M2[j][i];
				}
			}
		}
	} 
	void MultTransp(const matrix_t& M, const std::vector<double>& v, std::vector<double>& res) { //MT*v
		const std::size_t nrows = M.size();
		const std::size_t ncols = M.front().size();
		for (int i = 0; i < ncols; ++i) {
			res[i] = 0.0;
			for (int j = 0; j < nrows; ++j) {
				res[i] += M[j][i] * v[j];
			}
		}
	}
	void MultTransp(const matrix_t& M, const std::vector<double>& v, const std::vector<int>& activesetIndices, std::vector<double>& res) { 
		//MT*v on active set
		const std::size_t nrows = M.size();
		if (nrows == 0) {
			res.clear();
			return;
		}
		const std::size_t ncols = M.front().size();
		for (int i = 0; i < ncols; ++i) {
			res[i] = 0.0;
			for (int k = 0; k < nrows; ++k) {
				if (activesetIndices[k] == 1) {
					res[i] += M[k][i] * v[k];
				}
			}
		}
	}

    void MultTransp(const matrix_t& M, const std::vector<double>& v, const std::set<unsg_t>& activesetIndices, std::vector<double>& res) {
        //MT*v on active set
        const std::size_t nrows = M.size();
        if (nrows == 0) {
            res.clear();
            return;
        }
        const std::size_t ncols = M.front().size();
        std::fill(res.begin(), res.end(), 0.0);
        if (!activesetIndices.empty()) {
            for (std::size_t i = 0; i < ncols; ++i) {
                res[i] = 0.0;
                for (auto iAct: activesetIndices) {
                    res[i] += M[iAct][i] * v[iAct];
                }
            }
        }
    }


    void swapColumns(matrix_t& M, int c1, int c2) {
		if (c1 == c2){
			return;
		}
        const std::size_t n = M.size();
		if (n == 0) {
			return;
		}
		const std::size_t m = M.front().size();
		if (c1 >= m || c2 >= m) {
			return;
		}
		for (std::size_t r = 0; r < n; ++r) {
			std::swap(M[r][c1], M[r][c2]);
		}
	}

	void M1M2T(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT) {
		//MMT = M1 * M2T
		//m cols M1 = ncols M2
		const int nrM1 = M1.size();
		if (nrM1 == 0) {
			MMT.clear();
			return;
		}
		const int ncM1 = M1.front().size();
		if (ncM1 == 0) {
			MMT.clear();
			return;
		}
		const int nrM2 = M2.size();
		for (int i = 0; i < nrM1; ++i) {
			for (int j = 0; j < nrM2; ++j) {
				MMT[i][j] = 0.0;
				for (int k = 0; k < ncM1; ++k) {
					MMT[i][j] += M1[i][k] * M2[j][k];
				}
			}
		}
	}
	void M1TM2(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT) {
		//MMT = M1T * M2
		//n rows M1 = n rows M2
		const int nrM1 = M1.size();
		if (nrM1 == 0) {
			MMT.clear();
			return;
		}
		const int ncM1 = M1.front().size();
		if (ncM1 == 0) {
			MMT.clear();
			return;
		}
		const int nrM2 = M2.size();
		const int ncM2 = M2.front().size();
		for (int i = 0; i < ncM1; ++i) {
			for (int j = 0; j < ncM2; ++j) {
				MMT[i][j] = 0.0;
				for (int k = 0; k < nrM1; ++k) {
					MMT[i][j] += M1[k][i] * M2[k][j];
				}
			}
		}
	}
	void M2M1T(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT) {
		const matrix_t& M1_1 = M2;
		const matrix_t& M2_1 = M1;
		M1M2T(M1_1, M2_1, MMT);
	}
    void  InvertByGauss(const matrix_t& M, matrix_t& Minv) {
		matrix_t Mtmp = M;
        const int n = M.size(); // n is square matrix
        // Create the augmented matrix
        // Add the identity matrix
        // of order at the end of original matrix.
        for (int i = 0; i < n; i++) {
            Minv[i][i] = 1.0;
        }

        // Interchange the row of matrix,
        // interchanging of row will start from the last row
        for (int i = n - 1; i > 0; i--) {
            if (Mtmp[i - 1][0] < Mtmp[i][0]) {
                Mtmp[i].swap(Mtmp[i - 1]);
				Minv[i].swap(Minv[i - 1]);
            }
        }

        for (int i = 0; i < n; i++) {
			const double tmp = std::fabs(Mtmp[i][i]) < CONSTANTS::pivotZero ? CONSTANTS::pivotZero : Mtmp[i][i];
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    double temp = Mtmp[j][i] / tmp;
                    for (int k = 0; k < n; k++) {
						Minv[j][k] -= Minv[i][k] * temp;
						Mtmp[j][k] -= Mtmp[i][k] * temp;
                    }
                }
            }
        }

        // Multiply each row by a nonzero integer.
        // Divide row element by the diagonal element
        for (int i = 0; i < n; i++) {
            double temp = 1 / Mtmp[i][i];
            for (int j = 0; j < n; j++) {
                Mtmp[i][j] = Mtmp[i][j] * temp;
				Minv[i][j] = Minv[i][j] * temp;
            }
        }
	}

	// invert Lower Triangular matrix using Gauss method
	void InvertLTrByGauss(const matrix_t& M, matrix_t& Minv) {
		const int n = M.size();
		for (int i = 0; i < n; i++) {
			const double tmp = -1.0 / M[i][i];
			for (int j = 0; j < n; j++) {
				if (i == j) {
					Minv[i][j] = 1.0 / M[i][j]; // diagonal elements must be non-zero
				} else if (i > j) {
					double sum = 0;
					for (int k = j; k < i; k++) {
						sum += M[i][k] * Minv[k][j];
					}
					Minv[i][j] = sum * tmp;
				}
				else {
					Minv[i][j] = 0.0;
				}
			}
		}
	}

	void InvertTriangularFactorization(const matrix_t& M, matrix_t& Minv) {
		int iBg = 1;
		const int m = M.size();
		matrix_t N = M;
		Minv = -N; // N -> -N
		for (int i = 0; i < m; ++i) {
			N[i][i] = 0.0;
			Minv[i][i] = 1.0 / M[i][i];
		}
		matrix_t NP = N;
		matrix_t NNP = N;
		for (int i = 0; i < m - 2; ++i) {
			MultStrictLowTriangular(N, NP, NNP, 1, 1 + i);
			int k = 0;
			for (int ii = 2 + i; ii < m; ++ii) {
				int jEnd = ++k;
				for (int j = 0; j < jEnd; ++j) {
					Minv[ii][j] += NNP[ii][j];
				}
			}
			std::swap(NP, NNP);
		}
	}
#ifdef EIGEN
	void InvertEigen(const matrix_t& M, matrix_t& Inv) {
		const int m = M.size();
		Eigen::MatrixXd matrix(m, m);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				matrix(i, j) = M[i][j];
			}
		}
		matrix = matrix.inverse();
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				Inv[i][j] = matrix(i, j);
			}
		}	
	}
#endif
	void GetIdentityMatrix(int size, matrix_t& M) {
		M.resize(size, std::vector<double>(size, 0.0));
		for (int i = 0; i < size; ++i) {
			M[i][i] = 1.0;
		}
	}
	double bTAb(const std::vector<double>& b, const matrix_t& A) {
		return 0;
	}

	void InvertTriangle(const matrix_t& M, matrix_t& Minv) { // M^-1
		InvertByGauss(M, Minv);
	}

	const int MAX_N = 100;
	void pivot(matrix_t& a, std::vector<double> b, int n, int k) {
		int p = k;
		double max = fabs(a[k][k]);
		for (int i = k + 1; i < n; i++) {
			if (fabs(a[i][k]) > max) {
				max = fabs(a[i][k]);
				p = i;
			}
		}
		if (p != k) {
			for (int j = 0; j < n; j++) {
				double temp = a[k][j];
				a[k][j] = a[p][j];
				a[p][j] = temp;
			}
			double temp = b[k];
			b[k] = b[p];
			b[p] = temp;
		}
	}

	void invertLowerTriangular(matrix_t& a, int n) {
		std::vector<double> b(MAX_N);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					b[j] = 1.0;
				}
				else {
					b[j] = 0.0;
				}
			}
			pivot(a, b, n, i);
			for (int j = i + 1; j < n; j++) {
				double factor = a[j][i] / a[i][i];
				for (int k = i; k < n; k++) {
					a[j][k] -= factor * a[i][k];
				}
				b[j] -= factor * b[i];
			}
			for (int j = 0; j < i; j++) {
				b[j] = 0.0;
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				double sum = 0.0;
				for (int k = i; k < j; k++) {
					sum += a[j][k] * b[k];
				}
				b[j] -= sum;
			}
		}
		for (int i = 0; i < n; i++) {
			double temp = a[i][i];
			for (int j = 0; j < i; j++) {
				temp -= a[i][j] * b[j];
			}
			b[i] /= temp;
		}
		for (int i = n - 1; i >= 0; i--) {
			double temp = b[i];
			for (int j = i + 1; j < n; j++) {
				temp -= a[j][i] * b[j];
			}
			b[i] = temp / a[i][i];
		}
	}

	void Mult(const matrix_t& M, const std::vector<double>& v, std::vector<double>& res) { //M*v
		const int n = M.size();
		const int m = M.front().size();
		for (int i = 0; i < n; ++i) {
			res[i] = 0.0;
			for (int j = 0; j < m; ++j) {
				res[i] += M[i][j] * v[j];
			}
		}
	}

	void VSum(const std::vector<double>& v1, const std::vector<double>& v2, std::vector<double>& sum) { //v1+v2
		const int n = static_cast<int>(v1.size());
		for (int i = 0; i < n; ++i) {
			sum[i] = v1[i] + v2[i];
		}
	}
	void VAdd(std::vector<double>& v1, const std::vector<double>& v2) { // v1+=v2
		return;
	}
	double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
		double res = 0.0;
		const int sz = v1.size();
		for (int i = 0; i < sz; ++i) {
			res += v1[i] * v2[i];
		}
		return res;
	}
	double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2, const std::vector<int>& activeSetIndices) {
		double res = 0.0;
		const int sz = v1.size();
		for (int i = 0; i < sz; ++i) {
			if (activeSetIndices[i] == 1) {
				res += v1[i] * v2[i];
			}
		}
		return res;
	}
    double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2, const std::set<unsg_t>& activeSetIndices) {
        double res = 0.0;
        for (auto iAct: activeSetIndices) {
            res += v1[iAct] * v2[iAct];
        }
        return res;
    }
	void RRF(matrix_t & matrix) {
		int lead = 0;
		int rowCount = matrix.size();
		int colCount = matrix[0].size();
		for (int r = 0; r < rowCount; r++) {
			if (colCount <= lead) {
				return;
			}
			int i = r;
			while (matrix[i][lead] == 0) {
				i++;
				if (rowCount == i) {
					i = r;
					lead++;
					if (colCount == lead) {
						return;
					}
				}
			}
			swap(matrix[i], matrix[r]);
			double lv = matrix[r][lead];
			for (int j = 0; j < colCount; j++) {
				matrix[r][j] /= lv;
			}

			for (int i = 0; i < rowCount; i++) {
				if (i != r) {
					double lv = matrix[i][lead];
					for (int j = 0; j < colCount; j++) {
						matrix[i][j] -= lv * matrix[r][j];
					}
				}
			}
			lead++;
		}
	}
	void RRFB(matrix_t& matrix, std::vector<double>& b) {
		// computes reduced row form of matrix and b 
		int lead = 0;
		int rowCount = matrix.size();
		int columnCount = matrix[0].size();
		for (int r = 0; r < rowCount; r++) {
			if (lead >= columnCount) {
				return;
			}
			int i = r;
			while (matrix[i][lead] == 0) {
				i++;
				if (i == rowCount) {
					i = r;
					lead++;
					if (lead == columnCount) {
						return;
					}
				}
			}
			std::swap(matrix[i], matrix[r]);
			std::swap(b[i], b[r]);
			double lv = matrix[r][lead];
			for (int j = 0; j < columnCount; j++) {
				matrix[r][j] /= lv;
			}
			b[r] /= lv;
			for (int i = 0; i < rowCount; i++) {
				if (i != r) {
					double lv = matrix[i][lead];
					for (int j = 0; j < columnCount; j++) {
						matrix[i][j] -= lv * matrix[r][j];
					}
					b[i] -= lv * b[r];
				}
			}
			lead++;
		}
	}

	void RCFB(matrix_t& A, std::vector<double>& b) {
		const double EPSILON = 1e-10; 
		int n = A.size();
		int m = A[0].size();
		int lead = 0; // index of leading variable
		for (int r = 0; r < n; r++) {
			if (lead >= m) {
				break;
			}
			int i = r;
			while (abs(A[i][lead]) < EPSILON) {
				i++;
				if (i == n) {
					i = r;
					lead++;
					if (lead == m) {
						break;
					}
				}
			}
			if (lead < m) {
				std::swap(A[i], A[r]);
				double div = A[r][lead];
				for (int j = 0; j < m; j++) {
					A[r][j] /= div;
				}
				b[r] /= div;
				for (int k = 0; k < n; k++) {
					if (k != r) {
						double mult = A[k][lead];
						for (int j = 0; j < m; j++) {
							A[k][j] -= mult * A[r][j];
						}
						b[k] -= mult * b[r];
					}
				}
				lead++;
			}
		}
	}
	void          MultStrictLowTriangular(const matrix_t& M1, const matrix_t& M2, matrix_t& M1M2, int m1Bg, int m2Bg) {
		//Computes M1 * M2, where M1 and M2 are strict lower triangular matrices, m1Bg - first non-zero row in M1, m2Bg - first non-zero row in M2
		//M1M2 must be filled with zero in advance
		//M1 size  = M2 size
		const int m = M1.size();
		const int mMax = std::max(m1Bg, m2Bg);
		if (mMax >= m) {
			M1M2.clear();
		}
		for (int i = mMax + 1; i < m; ++i) {
			for (int j = 0; j < i - mMax  ; ++j) {
				M1M2[i][j] = 0.0;
				for (int k = j + mMax ; k < i; ++k) {
					M1M2[i][j] += M1[i][k] * M2[k][j];
				}
			}
		}
	}

	void PermuteColumns(matrix_t& A, const std::vector<int>& pmt) {
		const int n = pmt.size();
		for (int i = 0; i < n; ++i) {
			if (pmt[i] != -1) {
				swapColumns(A, i, pmt[i]);
			}
		}
	}

	void PTV(std::vector<double>& v, const std::vector<int>& pmt) {
        const int n = pmt.size();
		for (int i = 0; i < n; ++i) {
			if (pmt[i] != -1) {
				std::swap(v[i], v[pmt[i]]);
			}
		}
	}
}
