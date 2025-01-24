#include "utils.h"
#include <cmath>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Core>

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
        const std::size_t n = M.size();
        // tmp by eigen
        /*
        Eigen::MatrixXd A(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) = M[i][j];
            }
        }
        Eigen::MatrixXd L = A.llt().matrixL();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                cholF[i][j] = L(i, j);
            }
        }
        return true;
        */
		output.negativeBlocking = 1.0; 
		output.negativeDiag.clear();
		output.pivoting = false;

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
                        cholF[row][col] = sqrt(factor);
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
            if (std::fabs(Mtmp[i - 1][0]) < std::fabs(Mtmp[i][0])) {
                Mtmp[i].swap(Mtmp[i - 1]);
				Minv[i].swap(Minv[i - 1]);
            }
        }

        for (int i = 0; i < n; i++) {
            const double tmp = std::fabs(Mtmp[i][i]) < CONSTANTS::pivotZero ? 1.0 / CONSTANTS::pivotZero : 1.0 / Mtmp[i][i];
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    double temp = Mtmp[j][i] * tmp;
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
            const double temp = std::fabs(Mtmp[i][i]) < CONSTANTS::pivotZero ? 1.0 / CONSTANTS::pivotZero : 1.0 / Mtmp[i][i];
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

    void InvertHermit(const matrix_t& Chol, matrix_t& Inv) {
        //Inv must be allocated in advance
        //M = Chol_T * Chol, Chol - low triangular matrix
        const std::size_t n = Chol.size();
        for (int r = 0; r < n; ++r) {
            const double diagInv = 1.0 / Chol[r][r];
            for (int c = 0; c <= r; ++c) {
                Inv[r][c] = (c == r) ? diagInv : 0.0;
                for (int i = 0; i < r; ++i) {
                    Inv[r][c] -= Chol[r][i] * Inv[i][c];
                }
                Inv[r][c] *= diagInv;
                Inv[c][r] = Inv[r][c];
            }
        }
    }
    void InvertCholetsky(const matrix_t& Chol, matrix_t& Inv) {
        //Inv must be allocated with zeros in advance
        //M = Chol_T * Chol, Chol - low triangular matrix
        const std::size_t n = Chol.size();
        for (int r = 0; r < n; ++r) {
            Inv[r][r]  = 1.0 / Chol[r][r];
            for (int c = 0; c < r; ++c) {
                Inv[r][c] = 0.0;
                for (int i = c; i < r; ++i) {
                    Inv[r][c] -= Chol[r][i] * Inv[i][c];
                }
                Inv[r][c] *= Inv[r][r];
            }
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
    void  MultStrictLowTriangular(const matrix_t& M1, const matrix_t& M2, matrix_t& M1M2, int m1Bg, int m2Bg) {
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

    LDLT::LDLT(const matrix_t& M, const std::vector<double>& S):
        d(0.0), maxSize(M.size()), nX(M.front().size()), curIndex(0), actSize(0),
        M(M), S(S),
        L(matrix_t(maxSize, std::vector<double>(maxSize))),
        D(std::vector<double>(maxSize)),
        norms2(std::vector<double>(maxSize)),
        cache(matrix_t(maxSize, std::vector<double>(maxSize, inf)))
    {
        for (std::size_t r = 0; r < maxSize; ++r) {
            double sum = 0.0;
            for (std::size_t c = 0; c < nX ; ++c) {
                sum += M[r][c] * M[r][c];
            }
            norms2[r] = sum + S[r] * S[r];
        }
    }

    unsigned int LDLT::Compute(const std::set<unsigned int>& active) {
        for (auto iAct : active) {
            Add(iAct);
        }
        return ndzero;
    }

    void LDLT::Add(std::size_t rowNumber, bool delMode) {
        // add row with index rowNumber to last position and recompute L and D
        d = norms2[rowNumber];
        // check if row is already added. Note: in deleteMode it's OK
        bool exists = false;
        for (std::list<unsigned int>::iterator it = rows.begin(); it != rows.end(); ++it){
            if (*it == rowNumber) {
                exists = true;
                if (!delMode) {
                    std::cout << "LDLT Add() info: row " << rowNumber << " is already added" << std::endl;
                }
                break;
            }
        }
        if (!exists) {
            rows.push_back(rowNumber);
        }
        std::size_t i = 0;
        for (auto r : rows) {
            if (i >= actSize){
                break;
            }
            if (std::fabs(D[i]) < dTol) {
                L[actSize][i] = 0.0;
            } else {
                double dot = 0.0;
                const double cv = cache[rowNumber][r];
                if (isSame(cv, inf)) {
                    dot = S[r] * S[rowNumber];
                    for (size_t j = 0; j < nX; ++j) {
                        dot += M[r][j] * M[rowNumber][j];
                    }
                    cache[rowNumber][r] = dot;
                } else {
                    dot = cv;
                }
                for (size_t j = 0; j < i; ++j) {
                    dot -= L[i][j] * D[j] * L[actSize][j];
                }
                L[actSize][i] = dot / D[i];
                d -= L[actSize][i] * dot;
            }
            ++i;
        }
        if (d <= 0.0) {
            std::cout << "LDL warning: " << "d=" << d << " <= 0.0" << std::endl;
            d = 0.0;
            ++ndzero;
        }
        D[actSize] = d;
        L[actSize][actSize] = 1.0;
        ++actSize;
    }

    void LDLT::Delete(std::size_t rowNumber) {
        bool deleted = false;
        actSize = 0;
        for (std::list<unsigned int>::iterator it = rows.begin(); it != rows.end(); ++it){
            if (!deleted && (*it == rowNumber)) {
                if((it = rows.erase(it)) == rows.end()) {
                    return;
                }
                std::cout << "LDLT: deleted " << rowNumber << std::endl;
                deleted = true;
            }
            if (deleted) {
                Add(*it, true);
            } else {
                ++actSize;
            }
        }
    }

    void LDL::Set(const matrix_t& A) {
        this->A = A;
        dimR = static_cast<int>(A.size());
        dimC = static_cast<int>(A.front().size());
        L.resize(dimR, std::vector<double>(dimR, 0.0));
        D.resize(dimR, 0.0);
        l.resize(dimR, 0.0);
        curIndex = 0;
        d = 0.0;
    }

    void LDL::Compute() {
        L.front().front() = 1.0;
        D.front() = getARowNormSquared(0);
        curIndex = 1;
        while(curIndex < dimR) {
            compute_l();
            compute_d();
            update_L();
            update_D();
            ++curIndex;
        }
    }

    void LDL::Add(const std::vector<double>& row) {
        const int mSize = static_cast<int>(A.size());
        if (mSize == 0) {
            Set({row});
            Compute();
            return;
        }
        std::vector<double>b(mSize, 0.0); // b=A*rowT
        Mult(A, row, b);  // m*n2
        std::vector<double> l = b;
        solveLDb(b, l); // m2
        double dd = DotProduct(row, row);
        for (int i = 0; i < mSize; ++i) {
            dd -= l[i] * D[i] * l[i];
        }
        for (auto& r : L) {
            r.resize(mSize + 1, 0.0);
        }
        L.push_back(l);
        L.back().push_back(1.0);
        D.push_back(dd);
        A.push_back(row);
    }
    void LDL::Remove(int i) {
        A.erase(A.begin() + i);
        // if remove last row
        if (i == A.size()) {
            L.resize(i);
            D.resize(i);
            for (int j = 0; j < i; ++j) {
                L[j].resize(i);
            }
            return;
        }
        const double dd = D[i];
        const int n = static_cast<int>(L.size());
        // i=0...n-1
        const int nRowsMdd = n - i - 1; //n-1,...,1
        const int nColsMdd = nRowsMdd + 1;
        matrix_t Mdd(nRowsMdd, std::vector<double>(nColsMdd, 0.0));

        std::vector<double> droots(nRowsMdd);
        for (int j = 0; j < nRowsMdd; ++j) {
            double droot2 = 0.0;
            if (D[i + 1 + j] < 0.0) {
                std::cout << "LDL warning: " << "D[" << i + 1 + j << "]=" << D[1 + i + j] << "< 0" << std::endl;
                droot2 = 0.0;
            } else {
                droot2 = sqrt(D[i + 1 + j]);
            }
            droots[j] = droot2;
        }
        //fill L2*D2_1/2
        for (int ir = 0; ir < nRowsMdd; ++ir) {
            for (int ic = 0; ic < nRowsMdd; ++ic) {
                Mdd[ir][ic] = L[i + 1 + ir][i + 1 + ic] * droots[ic];
            }
        }
        //fill dd_1/2*L4
        if (dd < 0.0) {
            std::cout << "LDL warning: " << "D[" << i << "]=" << dd << "< 0" << std::endl;
        }
        const double ddSqrt = dd < 0.0 ? 0.0 : sqrt(dd);
        for (int ir = 0; ir < nRowsMdd; ++ir) {
            Mdd[ir][nRowsMdd] = ddSqrt * L[i + 1+ ir][i];
        }
        // solve L2_til * D2_til * L2_til
        LDL ldl;
        ldl.Set(Mdd);
        ldl.Compute();
        const matrix_t& Ltil = ldl.GetL();
        const std::vector<double>& Dtil = ldl.GetD();
        // update L,D with L_, D_
        update_L_remove(i, Ltil);
        D.resize(D.size() - 1);
        for (int j = 0; j < nRowsMdd; ++j) {
            D[j + i] = Dtil[j];
        }
    }
    void LDL::update_L_remove(int iRowDelete, const matrix_t& Ltil) {
        L.erase(L.begin() + iRowDelete);
        const int Lsize = static_cast<int>(L.size());
        for (int i = 0; i < Lsize; ++i) {
            L[i].resize(Lsize);
            if (i >= iRowDelete) {
                for (int j = 0; j < Lsize - iRowDelete; ++j) {
                    L[i][j + iRowDelete] = Ltil[i - iRowDelete][j];
                }
            }
        }
    }
    const matrix_t& LDL::GetL() {
        return L;
    }
    const std::vector<double>& LDL::GetD() {
        return D;
    }

    void LDL::compute_l() {
        //L_i * D_i * l_i+1 = A1:i * A_i+1T
        //b = A1:i * A_i+1T
        std::vector<double> b(curIndex, 0.0);
        for (int i = 0; i < curIndex; ++i) {
            for (int j = 0; j < dimC; ++j) {
                b[i] += A[i][j] * A[curIndex][j];
            }
        }
        solveLDb(b, l);
    }
    void LDL::compute_d() {
        d = getARowNormSquared(curIndex);
        for (int i = 0; i < curIndex; ++i) {
            d -= l[i] * D[i] * l[i];
        }
    }
    void LDL::update_L() {
        L[curIndex][curIndex] = 1.0;
        for (int i = 0; i < curIndex; ++i) {
            L[curIndex][i] = l[i];
        }
    }
    void LDL::update_D() {
        if (d <= 0.0) {
            std::cout << "LDL warning: " << "d=" << d << "<0" << std::endl;
            d = 0.0;
        }
        D[curIndex] = d;
    }
    double LDL::getARowNormSquared(int row) const {
        double norm2 = 0.0;
        for (int i = 0; i < dimC; ++i) {
            norm2 += A[row][i] * A[row][i];
        }
        return norm2;
    }
    void LDL::solveLDb(const std::vector<double>& b, std::vector<double>& l) {
        const int n = b.size();
        for (int i = 0; i < n; ++i) {
            if (std::fabs(D[i]) < 1.0e-20) {
                l[i] = 0.0;
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; ++j) {
                    sum += L[i][j] * D[j] * l[j];
                }
                l[i] = (b[i] - sum) / D[i]; // (b[i] - sum) / L[i][i] * D[i] , L[i][i] = 1
            }
        }
    }
    void LDLTM::Set(const matrix_t& M) {

    }
    void LDLTM::Compute() {

    }
    MmtLinSolver::MmtLinSolver(const matrix_t& M, const std::vector<double>& S):
        nDZero(0), maxSize(S.size()), curSize(0), gamma(1.0), ldlt(M,S), S(S),
        forward(std::vector<double>(maxSize)),
        backward(std::vector<double>(maxSize))
    {}
    unsigned int MmtLinSolver::Solve(const std::set<unsigned int>& active, double gamma) {
        nDZero = ldlt.Compute(active);
        curSize = active.size();
        this->gamma = gamma;
        Forward(active);
        Backward();
        return nDZero;
    }
    void MmtLinSolver::Forward(const std::set<unsigned int>& active) {
        const matrix_t& L = ldlt.GetL();
        std::size_t i = 0;
        for (auto iAct : active) {
            double sum = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                sum += L[i][j] * forward[j];
            }
            forward[i] = gamma * S[iAct] - sum;
            ++i;
        }
    }
    void MmtLinSolver::Backward() {
        const matrix_t& L = ldlt.GetL();
        const std::vector<double>& D = ldlt.GetD();
        for (int i = curSize - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < curSize; ++j) {
                sum += L[j][i] * D[i] * backward[j];
            }
            backward[i] = (std::fabs(D[i]) < zeroTol) ? 0.0 : (forward[i] - sum) / D[i];
        }
    }

    int MMTbSolver::Solve(const matrix_t& M, const std::vector<double>& b) {
        //solve MMTx=b
        assert(M.size() == b.size());
        LDL ldl;
        forward.resize(M.size());
        backward.resize(M.size());
        ldl.Set(M);
        ldl.Compute();
        ndzero = 0;
        std::vector<int> dzeroIndices(M.size(), -1);
        int j = 0;
        for (int i = 0; i < M.size(); ++i) {
            if (std::fabs(ldl.GetD()[i]) < zeroTol) {
                ndzero += 1;
                dzeroIndices[j++] = i;
            }
        }
        if (ndzero >= 0) {
            SolveForward(ldl.GetL(), b);
            SolveBackward(ldl.GetD(), ldl.GetL());
        } else {
            GetMMTKernel(dzeroIndices, ldl.GetL(), backward);
        }
        return ndzero;
    }
    void MMTbSolver::SolveForward(const matrix_t& L, const std::vector<double>& b) {
        const int n = b.size();
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += L[i][j] * forward[j];
            }
            forward[i] = b[i] - sum;
        }
    }
    void MMTbSolver::SolveBackward(const std::vector<double>& D, const matrix_t& L) {
        const int n = forward.size();
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum += L[j][i] * D[i] * backward[j];
            }
            backward[i] = std::fabs(D[i]) < zeroTol ? 0.0 : (forward[i] - sum) / D[i];
        }
    }
    void MMTbSolver::GetMMTKernel(const std::vector<int>& dzeroIndices, const matrix_t& L, std::vector<double>& ker) {

        if (ndzero > 0) { // last element
            const int zeroIndex = dzeroIndices.front();
            std::fill(ker.begin(), ker.end(), 0.0);
            const auto& row = L[zeroIndex];
            //solve backward Lx = -l
            for (int i = zeroIndex - 1; i >= 0; --i) {
                double sum = 0.0;
                for (int j = i + 1; j < zeroIndex; ++j) {
                    sum += L[j][i] * ker[j];
                }
                ker[i] = -row[i] - sum;
            }
            ker[zeroIndex] = 1.0;
        }

    }
    const std::vector<double>& MMTbSolver::GetSolution() {
        return backward;
    }
    int MMTbSolver::nDZero() {
        return ndzero;
    }

}
