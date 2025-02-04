#include "test_utils.h"
#include "utils.h"
#include "qp.h"
#include "data_writer.h"
#include "preprocessor.h"
#define PRP
bool isNumber(const double& val) {
	return std::isfinite(val);
}

double relativeVal(double a, double b) {
	const double zeroTol = 1.0e-15;
	if (isSame(a, 0.0, zeroTol) && isSame(b, 0.0, zeroTol)) {
		return zeroTol;
	}
	else if (isSame(a, 0.0, zeroTol)) {
		return std::fabs(a - b);
	}
	else {
		double diff = a - b;
		if (isSame(diff, 0.0, 1.0e-8)) {
			return zeroTol;
		}
		return std::fabs(diff / b);
	}
}
void TestMatrixMult(const matrix_t& m1, const matrix_t& m2, const matrix_t& baseline) {
    ASSERT_EQ(m1.front().size(), m2.size());
	matrix_t mult(m1.size(), std::vector<double>(m2.front().size(), 1.0)); // non-zero output
	Mult(m1, m2, mult);
	for (int i = 0; i < m1.size(); ++i) {
		for (int j = 0; j < m2.front().size(); ++j) {
			EXPECT_EQ(baseline[i][j], mult[i][j]);
		}
	}
}
void TestMatrixMultTranspose(const matrix_t& m, const std::vector<double>& v, const std::vector<double>& baseline) {
	ASSERT_EQ(m.size(), v.size());
	std::vector<double> res(m.front().size(), 1.0);
	MultTransp(m, v, res);
	for (std::size_t i = 0; i < m.front().size(); ++i) {
		EXPECT_EQ(res[i], baseline[i]);
	}
}
void TestMatrixMultStrictLowTriangular(const matrix_t& m1, const matrix_t& m2) {
	matrix_t resLT(m1.size(), std::vector<double>(m2.front().size(), 0.0));
	matrix_t resExpl = resLT;
	MultStrictLowTriangular(m1, m2, resLT, 1, 1);
	Mult(m1, m2, resExpl);
	const double eps = 1.0e-8;
	for (int i = 0; i < m1.size(); ++i) {
		for (int j = 0; j < m2.front().size(); ++j) {
			EXPECT_NEAR(resLT[i][j], resExpl[i][j], eps);
		}
	}
}
void TestInvertGauss(const matrix_t& m) {
	ASSERT_EQ(m.size(), m.front().size());
	matrix_t minv(m.size(), std::vector<double>(m.size(), 0.0));
	InvertByGauss(m, minv);
	matrix_t mult(m.size(), std::vector<double>(m.size()));
	Mult(m, minv, mult);
	const double eps = 1.0e-7;
	for (int i = 0; i < m.size(); ++i) {
		for (int j = 0; j < m.size(); ++j) {
			EXPECT_NEAR(mult[i][j], i == j ? 1.0 : 0.0, eps);
		}
	}
}
void TestInvertHermit(const matrix_t& m) {
    matrix_t cholF(m.size(), std::vector<double>(m.size(), 0.0));
    CholetskyOutput output;
    ComputeCholFactorT(m, cholF, output);
    matrix_t mInv = cholF;
    InvertHermit(cholF, mInv);
    matrix_t mult(m.size(), std::vector<double>(m.size()));
    Mult(m, mInv, mult);
    const double eps = 1.0e-7;
    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m.size(); ++j) {
            EXPECT_NEAR(mult[i][j], i == j ? 1.0 : 0.0, eps);
        }
    }
}
void TestInvertCholetsky(const matrix_t& m) {
    matrix_t cholF(m.size(), std::vector<double>(m.size(), 0.0));
    CholetskyOutput output;
    ComputeCholFactorT(m, cholF, output);
    matrix_t mInv(m.size(), std::vector<double>(m.size(), 0.0));
    InvertCholetsky(cholF, mInv);
    matrix_t mult(m.size(), std::vector<double>(m.size()));
    Mult(cholF, mInv, mult);
    const double eps = 1.0e-7;
    for (int i = 0; i < m.size(); ++i) {
        for (int j = 0; j < m.size(); ++j) {
            EXPECT_NEAR(mult[i][j], i == j ? 1.0 : 0.0, eps);
        }
    }
}
void TestLinearTransformation(const QP_NNLS_TEST_DATA::QPProblem& problem,
							  const matrix_t& trMatrix, const QP_NNLS_TEST_DATA::QPProblem& bl) {
	LinearTransform tr;
	tr.setQPProblem(problem);
	tr.transform(trMatrix);
	const matrix_t& At = tr.getA();
	const matrix_t& Ht = tr.getH();
	ASSERT_EQ(At.size(), problem.A.size());
	ASSERT_EQ(Ht.size(), problem.A.front().size());
	const double eqvTol = 1.0e-16;
	for (std::size_t i = 0; i < Ht.size(); ++i) {
		for (std::size_t j = 0; j < i; ++j) {
			EXPECT_NEAR(Ht[j][i], Ht[i][j], eqvTol);
			EXPECT_NEAR(Ht[j][i], bl.H[i][j], eqvTol);
		}
		EXPECT_NEAR(Ht[i][i], bl.H[i][i], eqvTol);
	}
	for (std::size_t i = 0; i < problem.A.size(); ++i) {
		for (std::size_t j = 0; j < Ht.size(); ++j) {
			EXPECT_EQ(At[i][j], bl.A[i][j]);
		}
	}
}
void TestM1M2T(const matrix_t& m1, const matrix_t& m2, const matrix_t& baseline) {
	ASSERT_EQ(m1.front().size(), m2.front().size());
	matrix_t m1m2T(m1.size(), std::vector<double>(m2.size()));
	M1M2T(m1, m2, m1m2T);
	const double tol = 1.0e-10;
	for (int i = 0; i < m1.size(); ++i) {
		for (int j = 0; j < m2.size(); ++j) {
			EXPECT_NEAR(m1m2T[i][j], baseline[i][j], tol);
		}
	}
}

void TestLDLT(LDLT& solver, const matrix_t& M,
              const std::vector<double>&S,
              const std::set<unsigned int>& active) {
    const std::size_t nR = M.size();
    const std::size_t nC = M.front().size();
    const std::size_t nActive = active.size();
    ASSERT_EQ(S.size(), nR);
    ASSERT_LE(nActive, nR);
    solver.Compute(active);
    const matrix_t& l = solver.GetL();
    const std::vector<double>& d = solver.GetD();
    matrix_t mmt(nActive, std::vector<double>(nActive));
    std::size_t i = 0;
    for (unsigned int iA : active) {
        std::size_t j = 0;
        for (unsigned int jA : active) {
            double sum = S[iA] * S[jA];
            for (std::size_t c = 0; c < nC; ++c) {
                sum += M[iA][c] * M[jA][c];
            }
            mmt[i][j++] = sum;
        }
        ++i;
    }
    matrix_t ld(nActive, std::vector<double>(nActive, 0.0));
    for (unsigned int i = 0; i < nActive; ++i) {
        for (unsigned int j = 0; j < nActive; ++j) {
            ld[i][j] = l[i][j] * d[j];
        }
    }
    matrix_t ldl(mmt);
    for (unsigned int i = 0; i < nActive; ++i) {
        for (unsigned int j = 0; j < nActive; ++j) {
            double sum = 0.0;
            for (std::size_t c = 0; c < nActive; ++c) {
                sum += ld[i][c] * l[j][c];
            }
            EXPECT_NEAR(mmt[i][j], sum, 1.0e-7);
        }
    }
}

void TestLDL(const matrix_t& M) {
	LDL ldl;
	ldl.Set(M);
	ldl.Compute();
	const matrix_t& L = ldl.GetL();
	const std::vector<double>& D = ldl.GetD();  
	ASSERT_EQ(L.size(), M.size());
	ASSERT_EQ(D.size(), M.size());
	// D has no zero elements and L[i][i] = 1;
	bool square = M.size() == M.front().size();
	for (int i = 0; i < M.size(); ++i) {
		//Di >= 0
		ASSERT_TRUE(std::fabs(D[i]) >= 0.0) << "D[i]=" << D[i] << " i=" << i;
	}
	// L is lower triangular
	for (int i = 0; i < M.size(); ++i) {
		for (int j = i + 1; j < M.size(); ++j) {
			ASSERT_TRUE(std::fabs(L[i][j]) < 1.0e-10) << "L[i][i]=" << L[i][i] << " i=" << i;
		}
	}
	matrix_t LD(M.size(), std::vector<double>(M.size(), 0.0));
	for (int i = 0; i < M.size(); ++i) {
		for (int j = 0; j < M.size(); ++j) {
			LD[i][j] = L[i][j] * D[j];
		}
	}
	matrix_t res(M.size(), std::vector<double>(M.size(), 0.0)); // LDLT
	matrix_t MMT(M.size(), std::vector<double>(M.size(), 0.0)); // MMT
	M1M2T(LD, L, res);
	M1M2T(M, M, MMT);
	//LDLT=MMT
	for (int i = 0; i < M.size(); ++i) {
		for (int j = 0; j < M.size(); ++j) {
			EXPECT_LT(relativeVal(MMT[i][j], res[i][j]), 1.0e-6) << "res=" << res[i][j] << " baseline="
					<< MMT[i][j] << " i-j " << i << " " << j;	
		}
	}
}

void TestLdlt(const matrix_t& M) {
    const std::size_t n = M.size();
    matrix_t L(n, std::vector<double>(n));
    std::vector<double> D(n);
    matrix_t P(n, std::vector<double>(n));
    Ldlt(M, L, D, P);
    ASSERT_EQ(L.size(), n);
    ASSERT_EQ(D.size(), n);
    ASSERT_EQ(P.size(), n);
    matrix_t LD(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            LD[i][j] = L[i][j] * D[j];
        }
    }
    matrix_t LDLT(n, std::vector<double>(n));
    M1M2T(LD, L, LDLT);
    matrix_t PTL(n, std::vector<double>(n));
    M1TM2(P, LDLT, PTL);
    Mult(PTL, P, LDLT);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_LT(relativeVal(M[i][j], LDLT[i][j]), 1.0e-6) << "res=" << LDLT[i][j] << " baseline="
                    << M[i][j] << " i-j " << i << " " << j;
        }
    }
}
void TestInPlaceLdlt(const matrix_t& M) {
    const std::size_t n = M.size();
    matrix_t MCP = M;
    std::vector<double> D(n);
    matrix_t P(n, std::vector<double>(n));
    size_t iZero = 2;
    size_t iNeg = 1;
    InPlaceLdlt(MCP, P, iZero, iNeg);
    ASSERT_LE(iZero, iNeg);
    matrix_t L(MCP);
    std::cout << "zn " << iZero << " " << iNeg << std::endl;
    std::cout << "d: ";
    for (std::size_t i = 0; i < n; ++i) {
        D[i] = L[i][i];
        std::cout << D[i] << " ";
        if (i < iZero) {
            EXPECT_GT(D[i], 0.0);
        } else if (i >= iZero && i < iNeg) {
            EXPECT_NEAR(D[i], 0.0, 1.0e-12);
        } else {
            EXPECT_LT(D[i], 0.0);
        }
        L[i][i] = 1.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            L[i][j] = 0.0;
        }
    }
    std::cout << std::endl;
    matrix_t LD(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            LD[i][j] = L[i][j] * D[j];
        }
    }
    matrix_t LDLT(n, std::vector<double>(n));
    M1M2T(LD, L, LDLT);
    matrix_t PTL(n, std::vector<double>(n));
    Mult(P, LDLT, PTL);
    M1M2T(PTL, P, LDLT);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_LT(relativeVal(M[i][j], LDLT[i][j]), 1.0e-6) << "res=" << LDLT[i][j] << " baseline="
                    << M[i][j] << " i-j " << i << " " << j;
        }
    }
}

void TestLDLRemove(matrix_t& M, int i) {
	LDL ldl;
	ldl.Set(M);
	ldl.Compute();
	ldl.Remove(i);
	const matrix_t& L_ = ldl.GetL();
	const std::vector<double>& D_ = ldl.GetD();
	const std::size_t L_size = M.size() - 1;
	ASSERT_EQ(L_.size(), L_size);
	ASSERT_EQ(D_.size(), L_size);
	matrix_t LD(L_size, std::vector<double>(L_size, 0.0));
	for (std::size_t i = 0; i < L_size; ++i) {
		for (std::size_t j = 0; j < L_size; ++j) {
			LD[i][j] = L_[i][j] * D_[j];
		}
	}
	M.erase(M.begin() + i);
	matrix_t res(L_size, std::vector<double>(L_size, 0.0)); // LDLT
	matrix_t MMT(L_size, std::vector<double>(L_size, 0.0)); // MMT
	M1M2T(LD, L_, res);
	M1M2T(M, M, MMT);
	//LDLT=MMT
	for (std::size_t i = 0; i < L_size; ++i) {
		for (std::size_t j = 0; j < L_size; ++j)  {
			EXPECT_LT(relativeVal(MMT[i][j], res[i][j]), 1.0e-6) << "res=" << res[i][j] << " baseline="
					<< MMT[i][j] << " i-j " << i << " " << j;
		}
	}
}


void TestLDLAdd(matrix_t& M, const std::vector<double>& vc) {
	LDL ldl;
	ldl.Set(M);
	ldl.Compute();
	ldl.Add(vc);
	const matrix_t& L = ldl.GetL();
	const std::vector<double>& D = ldl.GetD();
	ASSERT_EQ(L.size(), M.size() + 1);
	ASSERT_EQ(D.size(), M.size() + 1);
	M.push_back(vc);
	matrix_t MMT(M.size(), std::vector<double>(M.size(), 0.0));
	M1M2T(M, M, MMT);
	const std::size_t mSize = M.size();
	matrix_t LD(mSize, std::vector<double>(mSize, 0.0));
	for (std::size_t i = 0; i < mSize; ++i) {
		for (std::size_t j = 0; j < mSize; ++j) {
			LD[i][j] = L[i][j] * D[j];
		}
	}
	matrix_t res = LD;
	M1M2T(LD, L, res);
	for (std::size_t i = 0; i < mSize; ++i) {
		for (std::size_t j = 0; j < mSize; ++j) {
			EXPECT_LT(relativeVal(MMT[i][j], res[i][j]), 1.0e-6) << "res=" << res[i][j] << " baseline="
					<< MMT[i][j] << " i-j " << i << " " << j;
			}
		}
}



std::vector<double>  GenRandomVector(int dim, double lw, double up) {
	std::uniform_real_distribution<> dis(lw, up);
	std::vector<double> v(dim);
	for (int i = 0; i < dim; ++i) {
		v[i] = dis(gen);
	}
	return v;
}
matrix_t GenRandomMatrix(int nRows, int nCols, double lw, double up) {
	std::uniform_real_distribution<> dis(lw, up);
	matrix_t M(nRows, std::vector<double>(nCols));
	for (int i = 0; i < nRows; ++i) {
		for (int j = 0; j < nCols; ++j) {
			M[i][j] = dis(gen);
			if (i == j && std::fabs(M[i][j]) < 1.0e-7) {
				M[i][j] += 1.0e-7;
			}
		}
	}
	return M;
}
matrix_t GetRandomPosSemidefMatrix(int dim, double lw, double up) {
	matrix_t Mrand = GenRandomMatrix(dim, dim, lw, up);
	matrix_t MPS = Mrand;
	M1M2T(Mrand, Mrand, MPS);
	return MPS;
}
matrix_t genRandomStrictLowerTriangular(int mSize, int iBg) {
	const int lw = -100;
	const int up = 100;
	matrix_t Mrand = GenRandomMatrix(mSize, mSize, lw, up);
	int k = 0;
	for (int i = 0; i < mSize; ++i) {
		const int jBg = (i >= iBg) ? ++k : 0;
		for (int j = jBg; j < mSize; ++j) {
			Mrand[i][j] = 0.0;
		}
	}	
	return Mrand;
}

matrix_t GetRandomPosSemidefSymmetricMatrix(int dim, double lw, double up) {
	matrix_t dummy;
	return dummy;
}

void TestMMTb(const matrix_t& M, const std::vector<double>& b) {
	MMTbSolver solver;
	solver.Solve(M, b);
	const std::vector<double>& x = solver.GetSolution();
	ASSERT_EQ(x.size(), b.size());
	matrix_t MMT(M.size(), std::vector<double>(M.size()));
	M1M2T(M, M, MMT);
	std::vector<double> mmtx(b.size());
	Mult(MMT, x, mmtx);
	for (int i = 0; i < b.size(); ++i) {
		EXPECT_LT(std::fabs((b[i] - mmtx[i]) / b[i]), 1.0e-3) << "baseline=" << b[i] << " sol=" << mmtx[i] << " i=" << i;
	}
}

void TestSolverDense(const QP_NNLS_TEST_DATA::QPProblem& problem, const Settings& settings,
                     const QPBaseline& baseline, const std::string& logPath) {
    ProblemReader pr;
    pr.Init(problem.H, problem.c, problem.A, problem.b);
    QPNNLSDense solver;
    solver.SetCallback(std::make_unique<Callback1>(logPath));
    solver.Init(settings);
    ASSERT_TRUE(solver.SetProblem(pr.getProblem()));
    solver.Solve();
    const SolverOutput output = solver.GetOutput();
    ASSERT_EQ(output.dualExitStatus, baseline.dualStatus);
    ASSERT_EQ(output.primalExitStatus, baseline.primalStatus);
    const std::size_t nX = output.x.size();
    const auto& xBl = baseline.xOpt.front();
    ASSERT_EQ(nX, xBl.size());
    const double tol = 1.0e-5;
    EXPECT_LE(relativeVal(output.cost, baseline.cost), tol);
    for (std::size_t i = 0; i < nX; ++i) {
        EXPECT_LE(relativeVal(output.x[i], xBl[i]),tol);
    }
}

void LinearTransform::setQPProblem(const QP_NNLS_TEST_DATA::QPProblem& problem) {
	A = problem.A;
	b = problem.b;
	c = problem.c;
}

 const QP_NNLS_TEST_DATA::QPProblem& LinearTransform::transform(const matrix_t& trfMat) {
	trMat = trfMat;
	const int nConstraints = A.size();
	if (nConstraints == 0) {
		problemTr.clear();
		return problemTr;
	}
	const int nVariables = A.front().size();
	Ht.resize(nVariables, std::vector<double>(nVariables));
	At = A;
	ct = c;
	for (int i = 0; i < nVariables; ++i) {
		for (int j = 0; j <= i; ++j) {
			Ht[i][j] = 0.0;
			for (int k = 0; k < nVariables; ++k) {
				Ht[i][j] += trfMat[k][i] * trfMat[k][j];
			}
			if (i != j) {
				Ht[j][i] = Ht[i][j];
			}
		}
	}
	for (int i = 0; i < nConstraints; ++i) {
		for (int j = 0; j < nVariables; ++j) {
			At[i][j] = 0.0;
			for (int k = 0; k < nVariables; ++k) {
				At[i][j] += A[i][k] * trfMat[k][j];
			}
		}
	}
	for (int i = 0; i < nVariables; ++i) {
		ct[i] = 0.0;
		for (int j = 0; j < nVariables; ++j) {
			ct[i] += c[j] * trfMat[j][i];
		}
	}
	problemTr = QP_NNLS_TEST_DATA::QPProblem(Ht, At, ct, b);
	return problemTr;
}
const matrix_t& LinearTransform::getH() {
	return Ht;
}
const matrix_t& LinearTransform::getA() {
	return At;
}
const std::vector<double>& LinearTransform::getC() {
	return ct;
}
const std::vector<double>& LinearTransform::getInitCoordinates(const std::vector<double>& newCoordinates) {
	std::vector<double> oldCoordinates(Ht.size());
	Mult(trMat, newCoordinates,oldCoordinates);
	return oldCoordinates;
}

void LdltTester::Set(const matrix_t& M, const std::vector<double>& S) {
    this->M = M;
    this->S = S;
    nR = M.size();
    nC = M.front().size();
    solver = std::make_unique<LDLT>(M, S);
}
void LdltTester::Add(unsigned int index) {
    rows.insert(index);
    ASSERT_LT(index, nR);
    solver->Add(index);
    Check();
}
void LdltTester::AddPvt(unsigned int index) {
    rows.insert(index);
    ASSERT_LT(index, nR);
    solver->AddPvt(index);
    Check();
}
void LdltTester::Delete(unsigned int index) {
    rows.erase(index);
    ASSERT_LT(index, nR);
    solver->Delete(index);
    Check();
}
void LdltTester::Check() {
    const auto& rows = solver->GetRows();
    const std::size_t nr = rows.size();
    const auto pivots = solver->GetPivots();
    if (!pivots.empty()) {
        ASSERT_EQ(pivots.size(), nr);
        double pvt = std::numeric_limits<double>::max();
        for (auto el : pivots){
            ASSERT_LE(el, pvt);
            pvt = el;
        }
    }
    ASSERT_EQ(nr, this->rows.size());
    matrix_t mmt(nr, std::vector<double>(nr));
    std::size_t i = 0;
    for (unsigned int iR : rows) {
        std::size_t j = 0;
        for (unsigned int jR : rows) {
            double sum = S[iR] * S[jR];
            for (std::size_t c = 0; c < nC; ++c) {
                sum += M[iR][c] * M[jR][c];
            }
            mmt[i][j++] = sum;
        }
        ++i;
    }
    const matrix_t& l = solver->GetL();
    const std::vector<double>& d = solver->GetD();
    matrix_t ld(nr, std::vector<double>(nr, 0.0));
    for (unsigned int i = 0; i < nr; ++i) {
        for (unsigned int j = 0; j < nr; ++j) {
            ld[i][j] = l[i][j] * d[j];
        }
    }
    matrix_t ldl(mmt);
    for (unsigned int i = 0; i < nr; ++i) {
        for (unsigned int j = 0; j < nr; ++j) {
            double sum = 0.0;
            for (std::size_t c = 0; c < nr; ++c) {
                sum += ld[i][c] * l[j][c];
            }
            EXPECT_NEAR(mmt[i][j], sum, 1.0e-7);
        }
    }

}

void TestCholetskyBase::TestCholetsky(const matrix_t& m) {
	matrix_t A = m;
	M1M2T(m, m, A);
	matrix_t L(A.size(), std::vector<double>(A.size(), 0.0));
	ComputeCholFactor(A, L);
	matrix_t LLT = A;
	M1M2T(L, L, LLT);
	for (std::size_t i = 0; i < A.size(); ++i) {
		for (std::size_t j = 0; j < A.size(); ++j) {
			if (j > i) {
				EXPECT_NEAR(L[i][j], 0.0, zeroTol);
			}
			EXPECT_LE(relativeVal(LLT[i][j], A[i][j]), tol);
		}
	}
}

void TestCholetskyFullPivoting::TestCholetsky(const matrix_t& mInit) {
	matrix_t m = mInit; // save 
	matrix_t cholF(m.size(), std::vector<double>(m.size(), 0.0));
	std::vector<int> permut;
	int rk = QP_NNLS::ComputeCholFactorTFullPivoting(m, cholF, permut);
	ASSERT_EQ(rk, 0);
	matrix_t LTL = m;
	M1TM2(cholF, cholF, LTL);
	for (std::size_t i = 0; i < m.size(); ++i) {
		for (std::size_t j = 0; j < m.size(); ++j) {
			EXPECT_LE(relativeVal(LTL[i][j], m[i][j]), tol);
		}
	}
	// compute P_T * M * P
	std::cout << " permutation matrix: " << std::endl;
	for (std::size_t i = 0; i < permut.size(); ++i) {
		std::cout << permut[i] << " ";
		if (permut[i] >= 0) {
			std::swap(m[i], m[permut[i]]);
			swapColumns(m, i ,permut[i]);
		}
	}
	std::cout << std::endl;
	for (std::size_t i = 0; i < m.size(); ++i) {
		for (std::size_t j = 0; j < m.size(); ++j) {
			EXPECT_LE(relativeVal(mInit[i][j], m[i][j]), tol);
		}
	}
}

void QPSolverComparator::Set(QPSolvers solverType, CompareType compareType) {
	this->solverType = solverType;
	this->compareType = compareType;
}

void QPSolverComparator::Compare(const DenseQPProblem& problem, const Settings& settings, std::string logFile) {
	QP_SOLVERS::QPInput input;
	QP_SOLVERS::QPOutput output;
	input.m_H = problem.H;
	input.m_A = problem.A;
	input.m_bV = problem.b;
	input.m_cV = problem.c;
	input.m_x0 = std::vector<double>(problem.H.size(), 0.0); // initialize with zero, solver will change if need
	input.m_lower = problem.lw;
	input.m_upper = problem.up;
	if (solverType == QPSolvers::QLD) {
		solveQLD(input, output);
	} else {
		return;
	}
	if (output.exitStatus == 0) {
        std::cout << "CONSTRAINTS VIOLATIONS BASELINE BEGIN" << std::endl;
		CheckConstrViolations(input.m_A, input.m_bV, input.m_lower, input.m_upper, output.m_x);
        std::cout << "CONSTRAINTS VIOLATIONS BASELINE END" << std::endl;
	} 
    QP_NNLS::QPNNLSDense solver;
    solver.Init(QP_NNLS_TEST_DATA::NqpTestSettingsDefault);
    ASSERT_TRUE(solver.SetProblem(problem));
    solver.Solve();
    SolverOutput soutput = solver.GetOutput();
    QP_SOLVERS::QPOutput nnlsOutput;
    nnlsOutput.m_x = soutput.x;
    nnlsOutput.m_cost = soutput.cost;
    nnlsOutput.m_lambda = soutput.lambda;
    nnlsOutput.m_lambdaL = soutput.lambdaLw;
    nnlsOutput.m_lambdaU = soutput.lambdaUp;
    DualLoopExitStatus dualStatus = soutput.dualExitStatus;

    ASSERT_NE(dualStatus, DualLoopExitStatus::UNKNOWN);
    if (dualStatus != DualLoopExitStatus::INFEASIBILITY) {
        std::cout << "CONSTRAINTS VIOLATIONS TESTING BEGIN" << std::endl;
        CheckConstrViolations(problem.A, problem.b, problem.lw, problem.up, nnlsOutput.m_x);
        std::cout << "CONSTRAINTS VIOLATIONS TESTING END" << std::endl;
		nnlsOutput.exitStatus = 0;
	} else {
        nnlsOutput.exitStatus = 1;
	}
	ProcessResults(output, nnlsOutput);
}

void QPSolverComparator::CheckConstrViolations(const matrix_t& A, const std::vector<double>& b,
											   const std::vector<double>& lb, const std::vector<double>& ub, const std::vector<double>& x) {
	std::vector<double> Ax(A.size());
	Mult(A, x, Ax);
	// check ineq. constraints
	ASSERT_EQ(A.size(), b.size());
	const int nConstraints = A.size() - 2 * x.size() > 0 ? A.size() - 2 * x.size() : A.size();
	for (int i_c = 0; i_c < A.size(); ++i_c) {
		if (i_c < nConstraints) {
			double bWithViolation = std::numeric_limits<double>::min();
			if (isSame(b[i_c], 0.0, 1.0e-15)){
				bWithViolation = 1.0e-5;
			} else {
				//bWithViolation = b[i_c] > 0.0 ? b[i_c] * (1.0 + 1.0e-3) : b[i_c] * (1.0 - 1.0e-3);
				bWithViolation =  b[i_c] + 1.0e-7;
			}
			EXPECT_LE(Ax[i_c], bWithViolation);
		}
		else {
			ASSERT_TRUE( isSame(Ax[i_c], b[i_c], 1.0e-8) || Ax[i_c] < b[i_c]);
		}
	}
	// check bounds
	ASSERT_EQ(lb.size(), ub.size());
	const double factor = 1.0e-5;
	for (int i_b = 0; i_b < lb.size(); ++i_b) {
		ASSERT_TRUE(!isSame(lb[i_b], ub[i_b], 1.0e-14));
		double ubWithViolation = std::numeric_limits<double>::min();
	    double lbWithViolation = std::numeric_limits<double>::max();
		double dist = ub[i_b] - lb[i_b];
		ASSERT_GT(dist, 0.0);
		double tol = dist * factor;
		if (isSame(lb[i_b], 0.0, 1.0e-16)) {
			lbWithViolation = -tol;
			ubWithViolation = ub[i_b] * (1.0 + tol);
		} else if (isSame(ub[i_b], 0.0, 1.0e-16)) {
            lbWithViolation = lb[i_b] * (1.0 + tol);
			ubWithViolation = tol;
		} else if (ub[i_b] < 0.0) {
			lbWithViolation = lb[i_b] * (1.0 + tol);
			ubWithViolation = ub[i_b] * (1.0 - tol);
		} else if (lb[i_b] > 0.0) {
			lbWithViolation = lb[i_b] * (1.0 - tol);
			ubWithViolation = ub[i_b] * (1.0 + tol);
		} else if (lb[i_b] < 0.0 && ub[i_b] > 0.0) {
			lbWithViolation = lb[i_b] * (1.0 + tol);
			ubWithViolation = ub[i_b] * (1.0 + tol);
		} else {
			ASSERT_TRUE(false); 
		}
	}
}

void QPSolverComparator::ProcessResults(const QP_SOLVERS::QPOutput& outputSolver, const QP_SOLVERS::QPOutput& outputNNLS) {
	// first check exit statuses
	ASSERT_EQ(outputSolver.exitStatus, outputNNLS.exitStatus) << outputSolver.exitStatus << " " << outputNNLS.exitStatus;
	const std::vector<double> & x = outputSolver.m_x;
	const std::vector<double> & x_nnls = outputNNLS.m_x;
	const std::vector<double> & lambda = outputSolver.m_lambda;
	const std::vector<double> & lambda_nnls = outputNNLS.m_lambda;
	const std::vector<double> & lL = outputSolver.m_lambdaL;
	const std::vector<double> & lU = outputSolver.m_lambdaU;
	const std::vector<double> & lL_nnls = outputNNLS.m_lambdaL;
	const std::vector<double> & lU_nnls = outputNNLS.m_lambdaU;
	double cost = outputSolver.m_cost;
	double cost_nnls = outputNNLS.m_cost;
	// check sizes
	ASSERT_EQ(x.size(), x_nnls.size());
	ASSERT_EQ(lambda.size(), lambda_nnls.size());
	ASSERT_EQ(lU.size(),lL.size());
	ASSERT_EQ(lL_nnls.size(), lU_nnls.size());
	ASSERT_EQ(lU.size(), x.size());
	ASSERT_EQ(lU.size(), lU_nnls.size());
	const std::size_t nX = x.size();
	const std::size_t nLambda = lambda.size();
	// check if all values are in valid range
	ASSERT_TRUE(isNumber(outputSolver.m_cost) && isNumber(outputNNLS.m_cost));
    for (std::size_t iX = 0; iX < nX; ++iX) {
		ASSERT_TRUE(isNumber(x[iX]) && isNumber(x_nnls[iX]));
		ASSERT_TRUE(isNumber(lU[iX]) && isNumber(lU_nnls[iX]));
		ASSERT_TRUE(isNumber(lL[iX]) && isNumber(lL_nnls[iX]));
	}
	for (std::size_t iL = 0; iL < nLambda; ++iL) {
		ASSERT_TRUE(isNumber(lambda[iL]) && isNumber(lambda_nnls[iL]));
	}
    // compare with baseline
    if (compareType == CompareType::COST) {
        std::cout << "COMPARE COST" << std::endl;
		EXPECT_LE(cost_nnls, cost + costTol);
		return;
	}
	// compare x
	std::cout << "COMPARE X" << std::endl;
	CompareSequence(x_nnls, x);
	std::cout << "COMPARE LAMBDA CONSTRAINTS" << std::endl;
	CompareSequence(lambda_nnls, lambda);
	std::cout << "COMPARE LAMBDA LOWER BOUNDS" << std::endl;
    CompareSequence(lL_nnls, lL);
	std::cout << "COMPARE LAMBDA UPPER BOUNDS" << std::endl;
	CompareSequence(lU_nnls, lU);
}

void QPSolverComparator::CompareSequence(const std::vector<double>& seq, const std::vector<double>& bl) {
	ASSERT_EQ(seq.size(), bl.size());
	const std::size_t n = seq.size();
	for (std::size_t i = 0; i < n; ++i) {
		ASSERT_TRUE(isNumber(seq[i]) && isNumber(bl[i]));
		if (compareType == CompareType::RELATIVE || compareType == CompareType::STRICT) {
			CompareValue(seq[i], bl[i]);
		}
	}
}

void QPSolverComparator::CompareValue(double val, double bl) {
	if (compareType == CompareType::RELATIVE) {
		EXPECT_LE(relativeVal(val, bl), relTol) << "val / bl " << val << " " << bl;
	} else if (compareType == CompareType::STRICT) {
		EXPECT_NEAR(val, bl, strictTol) << "val / bl " << val << " " << bl;
	}
}

const QP_NNLS_TEST_DATA::QPProblem& HessianParametrizedTest::getProblem(const QP_NNLS_TEST_DATA::QPProblem& baseProblem) {
	problem = baseProblem;
	const std::size_t nx = baseProblem.c.size();
	const double diagFactor = GetParam();
	GetIdentityMatrix(nx, problem.H);
	if (modification == Modification::UNBALANCE) {
		problem.H.back().back() = diagFactor;
		for (int i = nx - 2; i >= 0; --i) {
			problem.H[i][i] = problem.H[i + 1][i + 1] * diagFactor;
		}
	} else if (modification == Modification::DIAG_SCALING) {
		for (std::size_t i = 0; i < nx; ++i) {
			problem.H[i][i] = diagFactor;
		}
	} else if (modification == Modification::STRATEGY_1) {
		problem.H.back().back() = diagFactor;
		for (int i = nx - 2; i >= 0; --i) {
			problem.H[i][i] = 1.0 + problem.H[i + 1][i + 1] * diagFactor * diagFactor;
		}
		for (std::size_t i = 0; i < nx; ++i) {
			for (std::size_t j = 0; j < i; ++j) {
				problem.H[i][j] = problem.H[j][i] = diagFactor;
			}
		}
	}
	return problem;
}


void ProblemReader::Init(const matrix_t& H, const std::vector<double>& c, const matrix_t& A, const std::vector<double>& b) {
	const std::size_t nVariables = c.size();
	if (H.empty()) {
        GetIdentityMatrix(nVariables, problem.H);
	} else {
		ASSERT_EQ(H.size(), nVariables);
		for (const auto& row: H) {
			ASSERT_EQ(row.size(), nVariables);
		}
		problem.H = H;
	}
	problem.c = c;
	const std::size_t nConstraints = A.size();
	if (nConstraints == b.size()) { // no bounds, set as +-maxValue
		problem.A = A;
		problem.b = b;
		problem.lw = std::vector<double>(nVariables, -1.0e19);
		problem.up = std::vector<double>(nVariables, 1.0e19);
	} else if (nConstraints < b.size() && nConstraints + 2 * nVariables == b.size()) {
		problem.b = std::vector<double>(b.begin(), b.begin() + nConstraints);
		problem.A = A;
		problem.lw.resize(nVariables, 0.0);
		problem.up.resize(nVariables, 0.0);
		for (std::size_t i = 0; i < nVariables; ++i) {
			problem.lw[i] = -b[nConstraints + 2 * i + 1];
			problem.up[i] = b[nConstraints + 2 * i];
			ASSERT_LT(problem.lw[i], problem.up[i]);
		}
	} else {
		ASSERT_TRUE(false) << "size of b < size of A: " << b.size() << " " << A.size();
	}
}
void ProblemReader::Init(const matrix_t& H, const std::vector<double>& c, const matrix_t& A, const std::vector<double>& b, const std::vector<double>& lw, const std::vector<double>& up) {
	    Init(H, c, A, b);
		ASSERT_EQ(H.size(), lw.size());
		ASSERT_EQ(H.size(), up.size());
		problem.lw = lw;
		problem.up = up;
	}

void QPBMComparator::Set(const CompareSettings& settings) {
	this->settings = settings;
}
void QPBMComparator::Compare(const QP_NNLS_TEST_DATA::QPProblem& problem, const std::string& logFile) {
	ProblemReader pr;
	pr.Init(problem.H, problem.c, problem.A, problem.b);
	DenseQPProblem qpProblem = pr.getProblem();
	QPSolverComparator qpComparator;
	qpComparator.Set(settings.qpSolver, settings.compareType);
    qpComparator.Compare(qpProblem, settings.uSettings);
}

void LinearTransformParametrized::SetUserSettings(const QP_NNLS::Settings& settings) {
	this->settings = settings;
}
void LinearTransformParametrized::TransformAndTest(const QP_NNLS_TEST_DATA::QPProblem& problem, const QPBaseline& baseline) {
    matrix_t trMat = GetParam();
	matrix_t trMatInv(trMat.size(), std::vector<double>(trMat.size(), 0.0));
	InvertByGauss(trMat, trMatInv);
	QPBaseline baselineTr = baseline;
	Mult(trMatInv, baseline.xOpt.front(), baselineTr.xOpt.front()); //Xnew = M-1*Xold
	LinearTransform tr;
	tr.setQPProblem(problem);
    TestSolverDense(tr.transform(trMat), QP_NNLS_TEST_DATA::NqpTestSettingsDefault, baselineTr, "testTransform.txt");
}
QPBaseline LinearTransformParametrized::ComputeBaseline(const QP_NNLS_TEST_DATA::QPProblem& problem) {
    QPNNLSDense solver;
	ProblemReader pr; 
	pr.Init(problem.H, problem.c, problem.A, problem.b);
	DenseQPProblem dProblem = pr.getProblem();
    solver.Init(settings);
    solver.SetProblem(dProblem);
    solver.Solve();
    SolverOutput output = solver.GetOutput();
    QPBaseline baseline;
    baseline.xOpt = {output.x};
    baseline.cost = output.cost;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
	return baseline;
}

void DenseQPTester::SetCoreSettings(const Settings& settings) {
    solver.Init(settings);
}

void DenseQPTester::SetCheckConditions(const QpCheckConditions& conditions) {

}

void DenseQPTester::SetUserCallback(std::unique_ptr<Callback> callback) {
    solver.SetCallback(std::move(callback));
}


const QPTestResult& DenseQPTester::Test(const DenseQPProblem& problem,
                                        const std::string& problemName) {

    this -> problemName = problemName;
    result.Reset();
    NqpPreprocessor prePr;
#ifdef PRP
    prePr.Prepare(problem);
    if(!solver.SetProblem(prePr.GetProblem())) {
#else
    if(!solver.SetProblem(problem)) {
#endif
        result.status = false;
        result.errMsg.clear();
        InitStageStatus initStatus = solver.GetInitStatus();
        if (initStatus == InitStageStatus::CHOLETSKY) {
            result.errMsg = "choletsky";
        } else if (initStatus == InitStageStatus::MATRIX_INVERSION) {
            result.errMsg = "chol inversion";
        }
    } else {
        solver.Solve();
        output = solver.GetOutput();
#ifdef PRP
        prePr.RecomputeQutput(output);
#endif
        CheckOutput(output);
        if (result.status) {
            ComputePrInfeasibility(problem);
            ComputeDlInfeasibility(problem);
            ComputeDualityGap(problem);
        }
    }
    FillReport();
    return result;
}
void DenseQPTester::CheckOutput(const SolverOutput& output) {
    result.nVariables = output.nVariables;
    result.nConstraints = output.nConstraints;
    result.nEqConstraints = output.nEqConstraints;
    result.nIterations = output.nDualIterations;
    if (output.dualExitStatus == DualLoopExitStatus::INFEASIBILITY) {
        result.status = false;
        result.errMsg = "infeasibility";
        return;
    }
    result.status = true;
}
void DenseQPTester::ComputePrInfeasibility(const DenseQPProblem& problem) {
    //Ax <= b, l <= x <= u
    //constraints
    double maxInfsblC = 0.0;
    unsigned int nViolatedC = 0;
    double violatedC = 0.0;
    for (std::size_t i = 0; i < problem.A.size(); ++i) {
        double constraint = -problem.b[i];
        for (std::size_t j = 0; j < problem.A[i].size(); ++j) {
            constraint += problem.A[i][j] * output.x[j];
        }
        if (constraint > maxInfsblC) {
            ++nViolatedC;
            violatedC = problem.b[i];
            maxInfsblC = constraint;
        }
    }
    //bounds
    double maxInfsblB = 0.0;
    unsigned int nViolatedB = 0;
    double violatedB = 0.0;
    for (std::size_t i = 0; i < problem.lw.size(); ++i) {
        double shiftU = output.x[i] - problem.up[i];
        double shiftL = problem.lw[i] - output.x[i];
        double shift = std::fmax(shiftL, shiftU);
        if (shift > maxInfsblB) {
            ++nViolatedB;
            violatedB = shiftL > shiftU ? problem.lw[i] : problem.up[i];
            maxInfsblB = shift;
        }
    }
    result.violatedC = violatedC;
    result.violatedB = violatedB;
    result.maxPrInfsbC = maxInfsblC;
    result.maxPrInfsbB = maxInfsblB;
    result.nPrInfsbC = nViolatedC;
    result.nPrInfsbB = nViolatedB;

}
void DenseQPTester::ComputeDlInfeasibility(const DenseQPProblem& problem) {
    // KKT conditions:
    // H * x + c + A_T * lambda - lambdaL + lambdaUp = 0 - dual feasibility
    // lambda, lambdaL, lambdaU >= 0
    // dualityGap = 0 <=> lambda *( A * x - b) = 0, lambdaL * (x - l) = 0, lambdaU * (x - u) = 0
    // A * x <= b, x <= u, x >= l -- checks in ComputePrInfeasibility

    // dual feasibility
    double maxDualFsb = 0.0;
    unsigned int nViolated = 0;   
    std::vector<double> dualFsb(problem.H.size());
    Mult(problem.H, output.x, dualFsb);
    std::vector<double> AtLambda(problem.H.size());
    MultTransp(problem.A, output.lambda, AtLambda);
    // xTx, xTx, bTl, lTl, uTl are duality gap components
    xHx = 0.0;
    cTx = 0.0;
    for (std::size_t i = 0; i < problem.H.size(); ++i) {
        xHx += dualFsb[i] * output.x[i];
        cTx += problem.c[i] * output.x[i];
        dualFsb[i] += (problem.c[i] - output.lambdaLw[i] +
                       output.lambdaUp[i] + AtLambda[i]);
        const double violation = std::fabs(dualFsb[i]);
        if (violation > 0.0) {
            maxDualFsb = std::fmax(violation, maxDualFsb);
            ++nViolated;
        }
    }
    result.maxDlInfsb = maxDualFsb;
    result.nDlInfsb = nViolated;

    // lambda > 0
    double maxNegDual = 0.0;
    unsigned int nNegLambda = 0;
    bTL = 0.0;
    for (std::size_t i = 0; i < output.nConstraints - 2 * output.nVariables; ++i) {
        bTL += problem.b[i] * output.lambda[i];
        if (output.lambda[i] < 0.0) {
            maxNegDual = std::fmax(maxNegDual, -output.lambda[i]);
            ++nNegLambda;
        }
    }
    lTL = 0.0;
    uTL = 0.0;
    for (std::size_t i = 0; i < output.nVariables; ++i) {
        lTL += output.lambdaLw[i] * problem.lw[i];
        uTL += output.lambdaUp[i] * problem.up[i];
        if (output.lambdaLw[i] < 0.0) {
            maxNegDual= std::fmax(maxNegDual, -output.lambdaLw[i]);
            ++nNegLambda;
        }
        if (output.lambdaUp[i] < 0.0) {
            maxNegDual = std::fmax(maxNegDual, -output.lambdaUp[i]);
            ++nNegLambda;
        }
    }
    result.maxNegDl = maxNegDual;
    result.nNegDl = nViolated;
}
void DenseQPTester::ComputeDualityGap(const DenseQPProblem& problem) {
    // gap = x_T * H * x + c_T * x + b_T * lambda - l_T * lambdaL + u_T * lambdaU
    result.dualityGap  = xHx + cTx + bTL - lTL + uTL;
}
void DenseQPTester::FillReport() {
    using namespace FMT_WRITER;
    if (logger.LineNumber() % 20 == 0) {
        logger.Write("test name",
                     "n variables",
                     "n constraints",
                     "n eq constraints",
                     "n iterations",
                     "status",
                     "max pr infsb cnstr",
                     "violated value",
                     "n violated",
                     "max pr infsb bnds",
                     "violated value",
                     "n violated",
                     "primal cost",
                     "duality gap",
                     "max dual infsb",
                     "max negative dual");
        logger.NewLine();
    }

    logger.Write(problemName,
                 result.nVariables,
                 result.nConstraints,
                 result.nEqConstraints,
                 result.nIterations);
    if (!result.errMsg.empty()) {
        logger.Write(result.errMsg);
    } else {
        logger.Write("solved",
                     result.maxPrInfsbC,
                     result.violatedC,
                     result.nPrInfsbC,
                     result.maxPrInfsbB,
                     result.violatedB,
                     result.nPrInfsbB,
                     output.cost,
                     result.dualityGap,
                     result.maxDlInfsb,
                     result.maxNegDl);
    }
    logger.NewLine();
}






