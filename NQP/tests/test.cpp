#include "utils.h"
#include "test_utils.h"
#include "test_data.h"
#include "TxtParser.h"
#include "decorators.h"
#include <algorithm>
#include <string>
#include "data_writer.h"
using namespace QP_NNLS;
using namespace QP_NNLS_TEST_DATA;
using namespace TXT_QP_PARSER;
const std::string TxtQpRoot = "C:/Users/m00829527/nqp/nqp/benchmarks/maros_meszaros_txt/Dense/noEq/";
#define ns(a) using namespace a;
TEST(DataWriter, Test1) {
    using namespace FMT_WRITER;
    FmtWriter fw;
    fw.SetFile("testLog.txt");
    fw.Write("header1", "header 2", "header 3", "header 4", "header      5");
    fw.Write(1.0, 2.0, 3.0, 4.0, 5.0);
    fw.Write(1, 2, 3);
    fw.Write(123423.23434, -234322.032424, 3.0, 44324423432.32442, 5.0);
    fw.Write(123.3432e10, -1.0e-6, 0.0, 1.0e20, 1.0e-20);
    fw.Write(1, -2, 3, 4, -900 ,10000, -10);
}
TEST(DataWriter, Test2) {
    using namespace FMT_WRITER;
    FmtWriter fw;
    fw.SetFile("testLog.txt");
    fw.Write(1, 2, 3, 4, 900 ,10000, -10);
    double x = 1.0;
    fw.Write(x);
    const std::string s = "abc";
    fw.Write(s);
    fw.Write(s, x);
}
TEST(DataWriter, Test3) {
    using namespace FMT_WRITER;
    FmtWriter fw;
    fw.SetFile("testLog3.txt");
    fw.Write(1, 2, 3, 4, 900 ,10000, -10);
    fw.NewLine();
    double x = 1.0;
    fw.Write(x);
    fw.NewLine();
    const std::string s = "abc";
    fw.Write(s);
    fw.Write(s, x);
    fw.NewLine();
}
TEST(TxtParserTests, QPTEST) {
    TxtParser parser;
    bool status = false;
    DenseQPProblem problem = parser.Parse(TxtQpRoot + "QPTEST.txt", status);
    ASSERT_TRUE(status);
    ASSERT_EQ(problem.H.size(), 2);
    for (const auto& row : problem.H) {
        ASSERT_EQ(row.size(), 2);
    }
    ASSERT_EQ(problem.A.size(), 4);
    for (const auto& row : problem.A) {
        ASSERT_EQ(row.size(), 2);
    }
    ASSERT_EQ(problem.c.size(), 2);
    ASSERT_EQ(problem.lw.size(), 4);
    ASSERT_EQ(problem.up.size(), 4);
    const double tol = 1.0e-8;
    EXPECT_NEAR(problem.H[0][0], 8.0, tol);
    EXPECT_NEAR(problem.H[0][1], 2.0, tol);
    EXPECT_NEAR(problem.H[1][0], 2.0, tol);
    EXPECT_NEAR(problem.H[1][1], 10.0, tol);
    EXPECT_NEAR(problem.A[0][0], 2.0, tol);
    EXPECT_NEAR(problem.A[0][1], 1.0, tol);
    EXPECT_NEAR(problem.A[1][0], -1.0, tol);
    EXPECT_NEAR(problem.A[1][1], 2.0, tol);
    EXPECT_NEAR(problem.A[2][0], 1.0, tol);
    EXPECT_NEAR(problem.A[2][1], 0.0, tol);
    EXPECT_NEAR(problem.A[3][0], 0.0, tol);
    EXPECT_NEAR(problem.A[3][1], 1.0, tol);
    EXPECT_NEAR(problem.c[0], 1.5, tol);
    EXPECT_NEAR(problem.c[1], -2.0, tol);
    EXPECT_NEAR(problem.lw[0], 2.0, tol);
    EXPECT_NEAR(problem.lw[1], -1.0e20, tol);
    EXPECT_NEAR(problem.lw[2], 0.0, tol);
    EXPECT_NEAR(problem.lw[3], 0.0, tol);
    EXPECT_NEAR(problem.up[0], 1.0e20, tol);
    EXPECT_NEAR(problem.up[1], 6.0, tol);
    EXPECT_NEAR(problem.up[2], 20.0, tol);
    EXPECT_NEAR(problem.up[3], 1.0e20, tol);
}

TEST(TxtParserTests, KSIP) {
    TxtParser parser;
    bool status = false;
    DenseQPProblem problem = parser.Parse(TxtQpRoot + "KSIP.txt", status);
    ASSERT_TRUE(status);
    const double tol = 1.0e-8;
    using namespace MAROS_MESZAROS;
    ASSERT_EQ(problem.H.size(), KSIP::H.size());
    ASSERT_EQ(problem.A.size(), KSIP::A.size());
    ASSERT_EQ(problem.c.size(), KSIP::c.size());
    ASSERT_EQ(problem.lw.size(), KSIP::lw.size());
    ASSERT_EQ(problem.up.size(), KSIP::up.size());
    const std::size_t nv = problem.H.size();
    for (std::size_t i = 0; i < nv; ++i) {
        ASSERT_EQ(problem.H[i].size(), KSIP::H[i].size());
        for (std::size_t j = 0; j < nv; ++j) {
            EXPECT_NEAR(problem.H[i][j], KSIP::H[i][j], tol);
        }
        EXPECT_NEAR(problem.c[i], KSIP::c[i], tol);
    }
    const std::size_t nc = problem.A.size();
    for (std::size_t i = 0; i < nc; ++i) {
        ASSERT_EQ(problem.A[i].size(), KSIP::A[i].size());
        for (std::size_t j = 0; j < nv; ++j) {
            EXPECT_NEAR(problem.A[i][j], KSIP::A[i][j], tol);
        }
        EXPECT_NEAR(problem.lw[i], KSIP::lw[i], tol);
        EXPECT_NEAR(problem.up[i], KSIP::up[i], tol) << i;
    }
}

TEST(TxtParserTests, HS118) {
    TxtParser parser;
    bool status = false;
    DenseQPProblem problem = parser.Parse(TxtQpRoot + "HS118.txt", status);
    ASSERT_TRUE(status);
    const double tol = 1.0e-8;
    using namespace MAROS_MESZAROS;
    ns(HS118)
    ASSERT_EQ(problem.H.size(), HS118::H.size());
    ASSERT_EQ(problem.A.size(), HS118::A.size());
    ASSERT_EQ(problem.c.size(), HS118::c.size());
    ASSERT_EQ(problem.lw.size(), HS118::lw.size());
    ASSERT_EQ(problem.up.size(), HS118::up.size());
    const std::size_t nv = problem.H.size();
    for (std::size_t i = 0; i < nv; ++i) {
        ASSERT_EQ(problem.H[i].size(), HS118::H[i].size());
        for (std::size_t j = 0; j < nv; ++j) {
            EXPECT_NEAR(problem.H[i][j], HS118::H[i][j], tol);
        }
        EXPECT_NEAR(problem.c[i], HS118::c[i], tol);
    }
    const std::size_t nc = problem.A.size();
    for (std::size_t i = 0; i < nc; ++i) {
        ASSERT_EQ(problem.A[i].size(), HS118::A[i].size());
        for (std::size_t j = 0; j < nv; ++j) {
            EXPECT_NEAR(problem.A[i][j], HS118::A[i][j], tol);
        }
        EXPECT_NEAR(problem.lw[i], HS118::lw[i], tol);
        EXPECT_NEAR(problem.up[i], HS118::up[i], tol) << i;
    }
}

TEST(Utils, MatrixMult1) {
	matrix_t M1 = { {1.0, 2.0}, {3.0, 4.0} };
	matrix_t baseline = { {7.0, 10.0}, {15.0, 22.0} };
	TestMatrixMult(M1, M1, baseline);
}
TEST(Utils, MatrixMult2) {
	//square 3x3
	matrix_t M1 = { {1.0, 2.0, -1.0}, {3.0, 4.0, -7.0}, {1.0, -5.0, 3.0} };
	matrix_t M2 = { {5.0, 2.0, -7.0}, {1.0, -2.0, -9.0}, {0, 6.0, -4.0} };
	matrix_t baseline = { {7.0, -8.0, -21.0}, {19.0, -44.0, -29.0}, {0.0 ,30.0, 26.0} };
	TestMatrixMult(M1, M2, baseline);
}
TEST(Utils, MatrixMult3) {
	//1x2  2x2
	matrix_t M1 = {{1.0, 2.0}};
	matrix_t M2 = {{3.0, 4.0}, {5.0, 6.0}};
	matrix_t baseline = {{13.0, 16.0}};
	TestMatrixMult(M1, M2, baseline);
}
TEST(Utils, MatrixMult4) {
	// 2x1  1x2
	matrix_t M1 = {{1.0}, {-5.0}};
	matrix_t M2 = {{3.0, -4.0}};
	matrix_t baseline = {{3.0, -4.0}, {-15.0, 20.0}};
	TestMatrixMult(M1, M2, baseline);
}
TEST(Utils, MultTransp1) {
	matrix_t M = {{1.0}, {2.0}};
	std::vector<double> v = {1.0, 2.0};
	std::vector<double> baseline = {5.0};
	TestMatrixMultTranspose(M, v, baseline);
}
TEST(Utils, MultTransp2) {
	matrix_t M = { {-5.0} };
	std::vector<double> v = {5.0};
	TestMatrixMultTranspose(M, v, {-25.0});
}
TEST(Utils, MultTransp3) {
	matrix_t M = { {1.0, 2.0}, {3.0, 4.0} };
	std::vector<double> v = { 5.0, -1.0};
	std::vector<double> baseline = { 2.0, 6.0 };
	TestMatrixMultTranspose(M, v, baseline);
}
TEST(Utils, MultTransp4) {
	matrix_t M = { {1.0, 2.0, 3.0}, {3.0, 4.0, 5.0} };
	std::vector<double> v = { 5.0, -1.0};
	std::vector<double> baseline = { 2.0, 6.0, 10.0 };
	TestMatrixMultTranspose(M, v, baseline);
}
TEST(Utils, MultTransp5) {
	matrix_t M = {{1.0, 2.0, 3.0, -2.0}, {3.0, 4.0, 5.0, 0.0}};
	std::vector<double> v = {5.0, -1.0};
	std::vector<double> baseline = {2.0, 6.0, 10.0, -10.0};
	TestMatrixMultTranspose(M, v, baseline);
}
TEST(Utils, MultStrictLowTriangular1) {
	const matrix_t M = { {0.0 ,0.0, 0.0}, {5.0, 0.0, 0.0},{ 7.0, 8.0 ,0.0} };
	TestMatrixMultStrictLowTriangular(M, M);
}
TEST(Utils, MultStrictLowTriangular2) {
	const matrix_t M1 = { {0.0 ,0.0, 0.0}, {5.0, 0.0, 0.0},{ 7.0, 8.0 ,0.0} };
	const matrix_t M2 = { {0.0 ,0.0, 0.0}, {-8.0, 0.0, 0.0},{ 21.0, 63.0 ,0.0} };
	TestMatrixMultStrictLowTriangular(M1, M2);
}
TEST(Utils, Randomized_MultStrictLowTriangular3) {
	TestMatrixMultStrictLowTriangular(genRandomStrictLowerTriangular(5,1), genRandomStrictLowerTriangular(5,1));
}
TEST(Utils, Randomized_MultStrictLowTriangular4) {
	const int mSize = 30;
	for (int shift = 1; shift < mSize - 2; ++shift) {
		TestMatrixMultStrictLowTriangular(genRandomStrictLowerTriangular(mSize, 1), 
		genRandomStrictLowerTriangular(mSize, shift));
	}
}

TEST(DISABLED_Utils, InvertTriangularByFactorization) {
	const int mMax = 100;
	for (int m = 10; m <= mMax; ++m) {
		matrix_t M = GetRandomPosSemidefMatrix(m, -10.0, 10.0);
		for (int i = 0; i < m; ++i) {
			for (int j = i; j < m; ++j) {
				M[i][j] = i == j ? 1.0 : 0.0;
			}
		}
		matrix_t Minv(m, std::vector<double>(m, 0.0));
		matrix_t MMinv(m, std::vector<double>(m, 0.0));
		InvertTriangularFactorization(M, Minv);
		Mult(M, Minv, MMinv);
		const double eps = 1.0e-8;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				EXPECT_NEAR( MMinv[i][j], i ==j ? 1.0 : 0.0, eps) << i << "/" << j;
			}
		}
	}
}
#ifdef EIGEN
TEST(Utils, InvertByEigenT1) {
	matrix_t M = { {5.0, 7.0},{8.0, 5.0} };
	const int m = M.size();
	matrix_t Minv(m, std::vector<double>(m, 0.0));
	matrix_t MMinv(m, std::vector<double>(m, 0.0));
	InvertEigen(M, Minv);
	Mult(M, Minv, MMinv);
	const double eps = 1.0e-8;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			EXPECT_NEAR(MMinv[i][j], i == j ? 1.0 : 0.0, eps) << i << "/" << j;
		}
	}
}
TEST(Utils, InvertLowDiagByEigen) {
	const int mMax = 7;
	for (int m = 7; m <= mMax; ++m) {
		matrix_t M = GetRandomPosSemidefMatrix(m, -10.0, 10.0);
		for (int i = 0; i < m; ++i) {
			for (int j = i; j < m; ++j) {
				M[i][j] = (i == j) ? 1.0 : 0.0;
			}
		}
		matrix_t Minv(m, std::vector<double>(m, 0.0));
		matrix_t MMinv(m, std::vector<double>(m, 0.0));
		InvertEigen(M, Minv);
		Mult(M, Minv, MMinv);
		const double eps = 1.0e-8;
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < m; ++j) {
				EXPECT_NEAR(MMinv[i][j], i == j ? 1.0 : 0.0, eps) << i << "/" << j;
			}
		}
	}
}
#endif

TEST(Utils, InvertByGauss1) {
	const matrix_t M = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
	TestInvertGauss(M);
}
TEST(Utils, Randomized_InvertByGauss2) {
	TestInvertGauss(GenRandomMatrix(3, 3, -1.0, 1.0));
}
TEST(Utils, Randomized_InvertByGauss3) {
	TestInvertGauss(GenRandomMatrix(3, 3, -100.0, 100.0));
}
TEST(Utils, Randomized_InvertByGauss4) {
	TestInvertGauss(GenRandomMatrix(10, 10, -1000.0, 1000.0));
}
TEST(Utils, InvertByGauss5) {
	TestInvertGauss({{0.5}});
}
TEST(Utils, Randomized_InvertByGaussBadConditionNumberLt10) {
	const int nVariablesMax = 10;
	for (int nVars = 1; nVars < nVariablesMax; ++nVars) {
		matrix_t M = GenRandomMatrix(nVars, nVars, -1000.0, 1000.0);
		M[0][0] = 1.0e-7;
		for (int i = 1; i < nVars; ++i) {
			M[i][i] = M[i - 1][i - 1] * 10.0 * (i % 2 == 0 ? 1.0 : -1.0);
		}
		TestInvertGauss(M);
	}
}
TEST(Utils, Randomized_InvertByGaussBadConditionNumber10_20) {
	const int nVariablesMax = 20;
	for (int nVars = 10; nVars < nVariablesMax; ++nVars) {
		matrix_t M = GenRandomMatrix(nVars, nVars, -1000.0, 1000.0);
		M[0][0] = 1.0e-7;
		for (int i = 1; i < nVars; ++i) {
			M[i][i] = M[i - 1][i - 1] * 10.0 * (i % 2 == 0 ? 1.0 : -1.0);
		}
		TestInvertGauss(M);
	}
}
TEST(Utils, Randomized_InvertByGaussBadConditionNumber20_40) {
	const int nVariablesMax = 40;
	for (int nVars = 20; nVars < nVariablesMax; ++nVars) {
		matrix_t M = GenRandomMatrix(nVars, nVars, -1000.0, 1000.0);
		M[0][0] = 1.0e-7;
		for (int i = 1; i < nVars; ++i) {
			M[i][i] = M[i - 1][i - 1] * 10.0 * (i % 2 == 0 ? 1.0 : -1.0);
		}
		TestInvertGauss(M);
	}
}

TEST(Utils, Randomized_InvertByGaussLowTriangle) {
	const int nVariablesMax = 100;
	for (int nVars = 100; nVars <= nVariablesMax; ++nVars) {
		matrix_t M = GenRandomMatrix(nVars, nVars, -100.0, 100.0);
		for (int i = 0; i < nVars; ++i) {
			for (int j = i + 1; j < nVars; ++j) {
				M[i][j] = 0.0;
			}
		}
		matrix_t Minv(M.size(), std::vector<double>(M.front().size(), 0.0));
		InvertLTrByGauss(M, Minv);
		const double eps = 1.0e-5;
		for (int i = 0; i < nVars; ++i) {
			for (int j = i + 1; j < nVars; ++j) {
				EXPECT_NEAR(Minv[i][j], 0.0, eps) << "Minv=" << Minv[i][j];
			}
		}
		matrix_t mult(M.size(), std::vector<double>(M.size()));
		Mult(M, Minv, mult);
		for (int i = 0; i < M.size(); ++i) {
			for (int j = 0; j < M.size(); ++j) {
				EXPECT_NEAR(mult[i][j], i == j ? 1.0 : 0.0, eps) << "mult=" << mult[i][j] << " "  << i << "/" << j;
			}
		}
	}
}
TEST(Utils, InvertHermitT1) {
    const matrix_t M = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    TestInvertHermit(M);
}
TEST(Utils, InvertHermitT2) {
    const matrix_t M = {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};
    TestInvertHermit(M);
}
TEST(Utils, InvertHermitT3) {
    const matrix_t M = {{3.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 1.0}};
    TestInvertHermit(M);
}
TEST(Utils, Randomized_InvertHermitT1) {
    TestInvertHermit(GetRandomPosSemidefMatrix(3, -1.0, 1.0));
}
TEST(Utils, Randomized_InvertHermitT2) {
    TestInvertHermit(GetRandomPosSemidefMatrix(10, -1.0, 1.0));
}
TEST(Utils, InvertCholetskyT1) {
    const matrix_t M = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    TestInvertCholetsky(M);
}
TEST(Utils, InvertCholetskyT2) {
    const matrix_t M = {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};
    TestInvertCholetsky(M);
}
TEST(Utils, InvertCholetskyT3) {
    const matrix_t M = {{-2.0, 0.0, 0.0, 0.0},
                        {6.0, 4.0, 0.0, 0.0},
                        {10.0, -1.0, -7.0, 0.0},
                        {3.0, -9.0, -2.0, 5.0}};
    matrix_t Inv = M;
    matrix_t mult = Inv;
    InvertCholetsky(M, Inv);
    Mult(M, Inv, mult);
    const double eps = 1.0e-7;
    for (int i = 0; i < M.size(); ++i) {
        for (int j = 0; j < M.size(); ++j) {
            EXPECT_NEAR(mult[i][j], i == j ? 1.0 : 0.0, eps);
        }
    }

}
TEST(Utils, Randomized_InvertCholetskyT1) {
    TestInvertCholetsky(GetRandomPosSemidefMatrix(10, -1.0, 1.0));
}
TEST(Utils, M1M2T_1) {
	const matrix_t M = { {1.0, 2.0}, {3.0, 4.0} };
	const matrix_t baseline = { {5.0, 11.0}, {11.0, 25.0} };
	TestM1M2T(M, M, baseline);
}
TEST(Utils, M1M2T_2) {
	const matrix_t M = { {-1.0, 2.0, -5.0}, {3.0, 4.0, -10.0}, {2.0 , 1.0, -1.0 } };
	const matrix_t baseline = { {30.0, 55.0, 5.0}, {55.0, 125.0, 20.0} ,{5.0, 20.0, 6.0} };
	TestM1M2T(M, M, baseline);
}
TEST(Utils, M1M2T_3) {
	const matrix_t M = {{5.0}, {7.0}};
	const matrix_t baseline = {{25.0, 35.0}, {35.0, 49.0}};
	TestM1M2T(M, M, baseline);
}
TEST(Utils, M1M2T_Simple4) {
	const matrix_t M1 = {{5.0}, {7.0}};
	const matrix_t M2 = {{5.0}, {7.0}, {9.0}};
	const matrix_t baseline = { {25.0, 35.0, 45.0}, {35.0, 49.0, 63.0} };
	TestM1M2T(M1, M2, baseline);
}
TEST(Utils, M1M2T_Simple5) {
	const matrix_t M1 = {{5.0}, {7.0}, {-1.0}};
	const matrix_t M2 = {{2.0}, {-5.0}, {9.0}};
	const matrix_t baseline = {{10.0, -25.0, 45.0}, {14.0, -35.0, 63.0}, {-2.0, 5.0, -9.0}};
	TestM1M2T(M1, M2, baseline);
}
TEST(Utils, M1M2T_Simple6) {
	const matrix_t M1 = {{5.0}};
	const matrix_t M2 = {{1.0}, {5.0}, {9.0}};
	const matrix_t baseline = {{5.0, 25.0, 45.0}};
	TestM1M2T(M1, M2, baseline);
}
TEST(Utils, M1M2T_Simple7) {
	const matrix_t M1 = {{5.0}, {-1.0}, {4.0}};
	const matrix_t M2 = {{1.0}};
	const matrix_t baseline = M1;
	TestM1M2T(M1, M2, baseline);
}
TEST(Utils, M1M2T_Simple8) {
	const matrix_t M1 = {{5.0, 3.0}, {-1.0, 2.0}, {4.0, -1.0}};
	const matrix_t M2 = {{1.0, 2.0}};
	const matrix_t baseline = {{11.0}, {3.0},{2.0}};
	TestM1M2T(M1, M2, baseline);
}
TEST_P(TestCholetskyParmetrizedRandom, Utils_Randomized_Cholesky) {
	Test(-1000.0, 1000.0);
}
INSTANTIATE_TEST_CASE_P(SUITE_1_SMALL,  TestCholetskyParmetrizedRandom, ::testing::Values(1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50));
INSTANTIATE_TEST_CASE_P(SUITE_2_SMALL,  TestCholetskyParmetrizedRandom, ::testing::Values(60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250));
#ifdef TEST_MEDIUM
INSTANTIATE_TEST_CASE_P(SUITE_3_MEDIUM, TestCholetskyParmetrizedRandom, ::testing::Values(300, 400, 500, 600, 700, 800, 900, 1000));
INSTANTIATE_TEST_CASE_P(SUITE_4_LARGE,  TestCholetskyParmetrizedRandom, ::testing::Values(2000, 3000, 4000, 5000));
#endif
TEST_P(TestCholetskyFullPivotingParametrized, Utils_Test1) {
    TestCholetsky(GetParam());
}
INSTANTIATE_TEST_CASE_P(SUITE_1, TestCholetskyFullPivotingParametrized, ::testing::Values(PSDM::mat1, PSDM::mat2, PSDM::mat3, PSDM::mat4, PSDM::mat5, PSDM::mat6));

TEST(Utils, InPlaceLDLT_T1) {
    matrix_t M = {{1.0, 0.0, 0.0},
                  {0.0, 1.0, 0.0},
                  {0.0, 0.0, 1.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T2) {
    matrix_t M = {{2.0, 0.0, 0.0},
                  {0.0, 2.0, 0.0},
                  {0.0, 0.0, 2.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T3) {
    matrix_t M = {{2.0, 1.0, 0.0},
                  {1.0, 2.0, 0.0},
                  {0.0, 0.0, 2.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T4) {
    matrix_t M = {{15.0, 10.0, 0.0},
                  {10.0, 2.0, 7.0},
                  {0.0, 7.0, 2.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T5) {
    matrix_t M = {{1.0, -1.0, 3.0},
                  {-1.0, 1.0, 0.0},
                  {3.0, 0.0, 1.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T6) {
    matrix_t M = {{1.0, -1.0, 3.0},
                  {-1.0, 1.0001, 0.0},
                  {3.0, 0.0, 1.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T10) {
    matrix_t M = {{2.0, -1.0, 3.0, 0.0},
                  {-1.0, 2.0, 0.0, 7.0},
                  {3.0, 0.0, -4.0, -5.0},
                  {0.0, 7.0, -5.0, -5.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_T11) {
    matrix_t M = {{1.0, -1.0, 3.0},
                  {-1.0, 1.0, 0.1},
                  {3.0, 0.1, 1.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_Swap2Elmts) {
    matrix_t M = {{1.0 , -1.0, 3.0, 7.0},
                  {-1.0, 1.0 , 0.1, -4.0},
                  {3.0 , 0.1 , 1.0, 1.0},
                  {7.0 ,-4.0 , 1.0, 10.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_1SwapT1) {
    //swap 0 and 3
    matrix_t M = {{1.0 , -1.0, 3.0, 7.0},
                  {-1.0, 1.0 , 0.1, -4.0},
                  {3.0 , 0.1 , 1.0, 1.0},
                  {7.0 ,-4.0 , 1.0, 10.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_1SwapT2) {
    //swap 0,3;
    matrix_t M = {{1.0 , -1.0, 3.0, 7.0},
                  {-1.0, 2.0 , 0.1, -4.0},
                  {3.0 , 0.1 , 1.0, 1.0},
                  {7.0 ,-4.0 , 1.0, 10.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_1SwapT3) {
    //swap 1,2;
    matrix_t M = {{ 5.0 , 0.1, -4.0},
                  { 0.1 , 1.0, 1.0},
                  {-4.0 , 1.0, 3.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_2SwapT1) {
    //swap 0,3; 1,2
    matrix_t M = {{1.0 , -1.0, 3.0, 7.0},
                  {-1.0, 1.0 , 0.1, -4.0},
                  {3.0 , 0.1 , 3.0, 1.0},
                  {7.0 ,-4.0 , 1.0, 10.0}};
    TestInPlaceLdlt(M);
}
TEST(Utils, InPlaceLDLT_PosSmdf1) {
    //swap 0,3; 1,2
    matrix_t M = {{1.0 , 2.0 , 3.0},
                  {2.0 , 2.0 , -2.0},
                  {3.0 ,-2.0 , 2.0}};

    TestInPlaceLdlt(M);
}
TEST(Utils_LDLT, Test1) {
    const matrix_t M = { {1.0}, {1.0} };
    const std::vector<double> S = {1.0, 1.0};
    const std::set<unsigned int> active = {0, 1};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test2) {
    const matrix_t M = { {1.0}, {2.0} };
    const std::vector<double> S = {3.0, 4.0};
    const std::set<unsigned int> active = {0, 1};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test3) {
    const matrix_t M = {{1.0}};
    const std::vector<double> S = {3.0};
    const std::set<unsigned int> active = {0};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test4) {
    const matrix_t M = {{1.0, 2.0, 3.0}, {-10.5, 100.7, 23.5}};
    const std::vector<double> S = {3.0, -400.987};
    const std::set<unsigned int> active = {0, 1};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test5) {
    const matrix_t M = {{1.0, 2.0, 3.0}, {-10.5, 100.7, 23.5}, {0.0, 0.0, 0.0}};
    const std::vector<double> S = {3.0, -400.987, 1.0};
    const std::set<unsigned int> active = {0, 1, 2};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test6) {
    const matrix_t M = {{1.0, 2.0, 3.0}, {-10.0, -10.0, -10.0}, {0.0, 0.0, 0.0}};
    const std::vector<double> S = {3.0, -10.0, 0.0};
    const std::set<unsigned int> active = {0, 1, 2};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test7) {
    const matrix_t M = {{1.0, 2.0, 3.0}, {-10.0, -10.0, -10.0}, {5.0, 5.0, 5.0}};
    const std::vector<double> S = {3.0, -10.0, 5.0};
    const std::set<unsigned int> active = {0, 1, 2};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test8) {
    const matrix_t M = {{1.0, 2.0, 3.0}, {-10.0, -10.0, -10.0}, {5.0, 5.0, 5.0}};
    const std::vector<double> S = {3.0, -10.0, 5.0};
    const std::set<unsigned int> active = {0, 2};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test9) {
    const matrix_t M = {{1.0, 2.0, 3.0},
                        {-10.0, -10.0, -10.0},
                        {5.0, 5.0, 5.0},
                        {0.002, 10.33, -25.9},
                        {-6.0, 10.0, -500.5 }};
    const std::vector<double> S = {3.0, -10.0, 5.0, 0.0, -9.0};
    const std::set<unsigned int> active = {0, 2, 4};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test10) {
    const matrix_t M = {{1.0, 2.0, -3.0},
                        {-10.0, -10.0, 10.0},
                        {5.0, 5.0, 5.0},
                        {-0.002, 10.33, 0.9},
                        {-6.0, 10.0, 1500.5 }};
    const std::vector<double> S = {3.0, -10.0, 15.0, 0.0, -9.0};
    const std::set<unsigned int> active = {0, 4};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Test11) {
    const matrix_t M = {{1.0, 2.0, -3.0},
                        {-10.0, -10.0, 10.0},
                        {5.0, 5.0, 5.0},
                        {-0.002, 10.33, 0.9},
                        {-6.0, 10.0, 1500.5 }};
    const std::vector<double> S = {3.0, -10.0, 15.0, 0.0, -9.0};
    std::set<unsigned int> active = {3};
    LDLT ldlt(M, S);
    TestLDLT(ldlt, M, S, active);
    active = {1, 4};
    TestLDLT(ldlt, M, S, active);
    active = {0, 2};
    TestLDLT(ldlt, M, S, active);
    active = {2, 3};
    TestLDLT(ldlt, M, S, active);
    active = {0};
    TestLDLT(ldlt, M, S, active);
}
TEST(Utils_LDLT, Add) {
    const matrix_t M = {{1.0, 2.0, -3.0},
                        {-10.0, -10.0, 10.0},
                        {5.0, 5.0, 5.0},
                        {-0.002, 10.33, 0.9},
                        {-6.0, 10.0, 1500.5 }};
    const std::vector<double> S = {3.0, -10.0, 15.0, 0.0, -9.0};
    LdltTester tester;
    tester.Set(M, S);
    tester.Add(1);
    tester.Add(3);
    tester.Add(2);
    tester.Add(0);
    tester.Add(4);
}
TEST(Utils_LDLT, Del) {
    const matrix_t M = {{1.0, 2.0, -3.0},
                        {-10.0, -10.0, 10.0},
                        {5.0, 5.0, 5.0},
                        {-0.002, 10.33, 0.9},
                        {-6.0, 10.0, 1500.5 }};
    const std::vector<double> S = {3.0, -10.0, 15.0, 0.0, -9.0};
    LdltTester tester;
    tester.Set(M, S);
    //case 1
    tester.Add(1);
    tester.Delete(1);
    //case 2
    tester.Add(2);
    tester.Add(0);
    tester.Delete(0);
    tester.Delete(2);
    //case 3          // matrix size:
    tester.Add(1);    // 1
    tester.Add(4);    // 2
    tester.Delete(1); // 1
    tester.Add(1);    // 2
    tester.Delete(4); // 1
    tester.Add(3);    // 2
    tester.Add(0);    // 3
    tester.Delete(3); // 2
    tester.Add(2);    // 3
}

TEST(Utils_LDLDecomposition, Test1) {
	matrix_t M = { {5.0, 11.0}, {11.0, 25.0} };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test2) {
	matrix_t M = { {5.5, 11.7}, {101.0, 1.234}, };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}

TEST(Utils_LDLDecomposition, Test3) {
	matrix_t M = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test4) {
	matrix_t M = { {5.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 7.0} };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test5) {
	matrix_t M = { {5.0, 1.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 7.0} };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test6) {
	matrix_t M = { {5.0, 1.0, 0.0}, {1.0, 2.0, -1.0}, {0.0, 0.0, 7.0} };
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test7) {
	matrix_t M = { {5.0, 1.0, 0.0, -2.24, 0.7},
		           {1.0, 2.0, -1.0, 213.9, -1.004},
		           {0.23, 0.0, 7.0, 11.233, 4.90 },
				   {7.44, 122.243, -5.0, 14.233, -2.1 },
				   {-5.83, 0.0, 113.463, 8.1733, 6.2 }
	};
	matrix_t MPS = M;
	M1M2T(M, M, MPS);
	TestLDL(MPS);
}
TEST(Utils_LDLDecomposition, Test8) {
	matrix_t M = { {5.0}, {11.0} };
	TestLDL(M);
}
TEST(Utils_LDLDecomposition, Test9) {
	matrix_t M = { {5.0}, {-11.0}, {16.423} };
	TestLDL(M);
}
TEST(Utils_LDLDecomposition, Test10) {
	matrix_t M = { {5.0}, {-11.0}, {16.423} };
	TestLDL(M);
}
TEST(Utils_LDLDecomposition, Test11) {
	matrix_t M = { {5.0, 7.0}, {-11.0, 2.5}, {-8.324, 24.2354} };
	TestLDL(M);
}
TEST(Utils_LDLDecomposition, Test12) {
	matrix_t M = {{5.5656}};
	TestLDL(M);
}
TEST(Utils_LDLDecomposition, Test13) {
	matrix_t M = {{5.5656, -100.34}};
	TestLDL(M);
}
TEST_P(TestLDLParametrized, Utils_Test1) {
	TestRandomMatrix(-100.0, 100.0); // matrix for test will be M * M_T
}
TEST_P(TestLDLParametrized, Utils_Test2) {
	TestRandomMatrix(-1000.0, 1000.0); // matrix for test will be M * M_T
}
TEST(Utils_LDLDecomposition, RemoveIdentity) {
	matrix_t M = { {1.0, 0.0, 0.0} , {0.0, 1.0, 0.0} ,{0.0, 0.0, 1.0} };
	TestLDLRemove(M, 1);
}
TEST(Utils_LDLDecomposition, RemoveSimple) {
	matrix_t M = { {1.0},{2.0},{3.0} };
	TestLDLRemove(M, 1);
}
TEST(Utils_LDLDecomposition, Remove1RowMatrix1) {
	matrix_t M = { {1.0} };
	TestLDLRemove(M, 0);
}
TEST(Utils_LDLDecomposition, Remove1RowMatrix2) {
	matrix_t M = { {1.0}, {2.0} };
	TestLDLRemove(M, 0);
}
TEST(Utils_LDLDecomposition, RemoveLast) {
	matrix_t M = { {1.0, 2.0, 3.0, 8.5, 3.6}, {2.0, 3.0, 4.0, -11.0, -1.7},
				   {24.0 ,5.0, 6.0, 7.0, 8.0}, {0.5, -12.0, 4.0, 9.0, 0.5},
				   {5.0 ,2.0, 6.0, 4.0, 8.0} };
	TestLDLRemove(M, M.size() - 1);
}
TEST(Utils_LDLDecomposition, RemoveZeroLine) {
	matrix_t M = { {1.0, 2.0, 3.0, 8.5, 3.6}, {2.0, 3.0, 4.0, -11.0, -1.7},
				   {24.0 ,5.0, 6.0, 7.0, 8.0}, {0.5, -12.0, 4.0, 9.0, 0.5},
				   {5.0 ,2.0, 6.0, 4.0, 8.0} };
	TestLDLRemove(M, 0);
}
//Generate random matrices and remove rows
TEST_P(TestLDLParametrized, Utils_TestRemoveSquare) {
	TestRemoveFromSquareRandomMatrix(-100.0, 100.0);
}
TEST_P(TestLDLParametrized, Utils_TestRemoveRectRGtC) {
	TestRemoveFromRectRandomMatrix(-100.0, 100.0); // n rows >= n cols
}
TEST_P(TestLDLParametrized, Utils_TestRemoveRectCGtR) {
	TestRemoveFromRectRandomMatrix(-100.0, 100.0, false); // n cols >= n rows
}
TEST(LDLDecomposition, AddSimple1) {
	matrix_t M = { {1.0, 0.0, 0.0} };
	const std::vector<double> toAdd = { 0.0, 1.0, 0.0 };
	TestLDLAdd(M, toAdd);
}
TEST(LDLDecomposition, AddSimple2) {
	matrix_t M = { {1.0, 0.0, 3.0} };
	const std::vector<double> toAdd = { -2.5, 1.0, 4.5 };
	TestLDLAdd(M, toAdd);
}
TEST_P(TestLDLParametrized, Utils_TestAddRGtC) {
	TestAdd(-100.0, 100.0); // n cols >= n rows
}
TEST_P(TestLDLParametrized, Utils_TestAddCGtR) {
	TestAdd(-100.0, 100.0, false); // n cols >= n rows
}

constexpr std::size_t n1Bg = 1;
constexpr std::size_t n1End = 10;
constexpr std::size_t n1step = 1;
constexpr std::size_t n2Bg = 10;
constexpr std::size_t n2End = 100;
constexpr std::size_t n2step = 10;
INSTANTIATE_TEST_CASE_P(SUITE_1, TestLDLParametrized, ::testing::Range(n1Bg, n1End, n1step));
INSTANTIATE_TEST_CASE_P(SUITE_2, TestLDLParametrized, ::testing::Range(n2Bg, n2End, n2step));

TEST(Utils_SolveMMTb, Dummy) {
	matrix_t M = { {-5.0} };
	std::vector<double> b = { 1.0 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Vector) {
	matrix_t M = { {-5.0, 3.0, -1.0} };
	std::vector<double> b = { 1.0 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Test1) {
	matrix_t M = { {1.0, 0.0} , {0.0, 1.0} };
	std::vector<double> b = { 2.0, 2.0 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Test2) {
	matrix_t M = { {1.0, 0.0, 0.0} , {0.0, 1.0, 0.0} ,{0.0, 0.0, 1.0} };
	std::vector<double> b = { 1.0, 2.0, 3.0 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Test3) {
	matrix_t M = { {5.0, 1.0} , {3.0, -1.0} };
	std::vector<double> b = { 1.0, 2.0};
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Test4) {
	matrix_t M = { {1.0, 2.0, 3.0} , {-1.0, 6.0, 4.0} };
	std::vector<double> b = { -2.0, 2.0 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, Test5) {
	matrix_t M = { {5.0, 1.0, 0.0, -2.24, 0.7},
				   {1.0, 2.0, -1.0, 213.9, -1.004},
				   {0.23, 0.0, 7.0, 11.233, 4.90},
				   {7.44, 122.243, -5.0, 14.233, -2.1},
				   {-5.83, 0.0, 113.463, 8.1733, 6.2}
				};
	std::vector<double> b = { -1.0, 3.0, 125.74324, 0.8414, 2.16 };
	TestMMTb(M, b);
}
TEST(Utils_SolveMMTb, RandomizedTestMain) {
	const int nRuns = 30;
	for (int i = 1; i <= nRuns; ++i) {
		double lw = -5.0 * (i + 1);
		double up = -lw;
		const matrix_t M = GetRandomPosSemidefMatrix(i, lw, up);
		matrix_t MPS = M;
		M1M2T(M, M, MPS);
		const std::vector<double> b = GenRandomVector(i, lw, up);
		TestMMTb(M, b);
	}
}
// Linear transformation of problem
// x_T * H * x + c_T * x ; A * x < b  x = Tr * x_new
// x_new_T * H_new * x_new + (Tr * c)_T * x_new
// The idea is to generate random problem from with identity hessian with same cost value
TEST(Utils_QPLinearTransform, Test1) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.H = {{1.0, 0.0}, {0.0, 1.0}};
	problem.A = {{1.0, 2.0}, {3.0, 4.0}};
	problem.b = { 1.0, 2.0 };
	problem.c = {0.0, 0.0};
	const matrix_t trMat = {{1.0, 0.0}, {0.0, 1.0}};
	QP_NNLS_TEST_DATA::QPProblem bl = problem;
	TestLinearTransformation(problem, trMat, bl);
}
TEST(Utils_QPLinearTransform, Test2) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.H = {{1.0, 0.0}, {0.0, 1.0}};
	problem.A = {{1.0, 2.0}, {3.0, 4.0}};
	problem.b = {1.0, 2.0};
	problem.c = {1.0, -1.0};
	matrix_t trMat = {{2.0, -1.0} , {3.0, 4.0}};
	QP_NNLS_TEST_DATA::QPProblem bl;
	bl.H = {{13.0 , 10.0}, {10.0, 17.0}};
	bl.A = {{8.0, 7.0}, {18.0, 13.0}};
	bl.c = {-1.0, -3.0};
	bl.b = problem.b;
	TestLinearTransformation(problem, trMat, bl);
}
TEST(Utils_QPLinearTransform, Test3) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.A = {{1.0, 2.0, -7.0}, {3.0, 4.0, -1.0}};
	problem.b = {1.0, 2.0};
	GetIdentityMatrix(3, problem.H);
	problem.c = {1.0, 2.0, 3.0};
	const matrix_t trMat = {{2.0, -1.0, 4.0}, {3.0, 4.0, -6.0}, {1.0, 5.0, -1.0}};
	QP_NNLS_TEST_DATA::QPProblem bl;
	bl.H = {{14.0, 15.0, -11.0}, {15.0, 42.0, -33.0}, {-11.0, -33.0, 53.0}};
	bl.A = {{1.0, -28.0, -1.0}, {17.0 , 8.0, -11.0}};
	bl.b = problem.b;
	bl.c = {11.0, 22.0, -11.0};
	TestLinearTransformation(problem, trMat, bl);
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest1) {
	QPBaseline baseline;
	baseline.xOpt = {{-0.5, 0.5}};
	baseline.cost = 0.25;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_2, NqpTestSettingsDefault, baseline, "test1.txt");
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest2) {
	QPBaseline baseline;
	baseline.xOpt = {{0.5, 1.5}};
	baseline.cost = 1.25;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_3, NqpTestSettingsDefault, baseline, "test2.txt");
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest3) {
	QPBaseline baseline;
	baseline.xOpt = {{0.0, 1.0}};
	baseline.cost = 0.5;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_4, NqpTestSettingsDefault, baseline, "test3.txt");
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest4) {
	QPBaseline baseline;
	baseline.xOpt = { {1.5, 2.5} };
	baseline.cost = 4.25;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_5, NqpTestSettingsDefault, baseline, "test4.txt");
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest5) {
	QPBaseline baseline;
	baseline.xOpt = {{0, 1.0}};
	baseline.cost = 0.5;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_6, NqpTestSettingsDefault, baseline, "test5.txt");
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest6) {
	QPBaseline baseline;
	baseline.xOpt = { {-0.5, 1.5} };
	baseline.cost = 1.25;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_7, NqpTestSettingsDefault, baseline, "test7.txt");
}
TEST_P(HessianParametrizedTest, SolutionOnConstraintsDiagHessSameValues) {
	QPBaseline baseline;
	baseline.xOpt = {{-0.5, 1.5}};
	baseline.cost = 1.25 * GetScaleFactor();
	SetModification(HessianParametrizedTest::Modification::DIAG_SCALING);
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(getProblem(case_7), NqpTestSettingsDefault, baseline, "test8.txt");
}
TEST_P(LinearTransformParametrized, Solver_T1) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.H = {{1.0 , 0.0}, {0.0, 1.0}};
	problem.c = { 0.0, 0.0 };
	problem.A = { {-1.0, -1.0}, {2.0 , -1.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0} };
	problem.b = { -1.0, -1.0, 1.5, 0.0, -0.5 };
	QPBaseline baseline;
	baseline.cost = 1.25;
	baseline.xOpt = {{-0.5, 1.5}};
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
	SetUserSettings(NqpTestSettingsDefault);
	TransformAndTest(problem, baseline);
}
TEST_P(LinearTransformParametrized, Solver_T2) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.H = {{1.0 , 0.0}, {0.0, 1.0}};
	problem.c = {3.0, 1.25};
	problem.A = {{-1.0, -1.0}, {2.0, -1.0}, {0.0, 1.0}, {1.0, 0.0}};
	problem.b = {-1.0, -1.0, 1.5, -0.5};
	SetUserSettings(NqpTestSettingsDefault);
	TransformAndTest(problem, ComputeBaseline(problem));
}
TEST_P(LinearTransformParametrized, Solver_T3) {
	QP_NNLS_TEST_DATA::QPProblem problem;
	problem.H = {{1.0 , 0.0}, {0.0, 1.0}};
	problem.c = {3.0, 1.25};
	problem.A = { {-1.0, -1.0}, {2.0 , -1.0}, {0.0, 1.0}, {1.0, 0.0}, {-2.0, -2.0}, {0.0, 5.0} };
	problem.b = { -1.0, -1.0, 1.5, -0.5, -2.0, 7.5 };
	SetUserSettings(NqpTestSettingsDefault);
	TransformAndTest(problem, ComputeBaseline(problem));
}

using namespace TRANSFORMATION_MATRIX;
INSTANTIATE_TEST_CASE_P(SUITE_1, LinearTransformParametrized, ::testing::Values(trMat1, trMat2, trMat3, trMat4, trMat5));

TEST_P(CompareRelHessParametrized, Solver_SolOnConstraintsT1) {
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	comparator.Compare(getProblem(simple_1), "case_on_constraints_prm1_" + std::to_string(static_cast<int>(GetParam())) + ".txt");
}

TEST_F(QPTestRelative, Solver_SolOnConstraintsT2) {
	comparator.Compare(simple_2, "case_on_constraints_2.txt");
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsT1) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(getProblem(case_18), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsT2) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(getProblem(case_19), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(CompareRelHessParametrized, Solver_SolNotOnConstraintsParametrizedT1) {
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    Settings settings = NqpTestSettingsDefault;
    //settings.cholPvtStrategy = CholPivotingStrategy::FULL;
	this->settings.uSettings = settings;
	QPProblem problem;
	problem.c = { 2.0, 4.3 };
	problem.A = { { 0.0, 1.0 }, {0.0, -1.0}, {1.0, 0.0}, {-1.0, 0.0} };
	problem.b = { 1.0, 1.0, 0.5, 2.5 };
	comparator.Compare(getProblem(problem), "case_not_on_constraints_prm1_" + std::to_string(static_cast<int>(GetParam())) + ".txt");
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsNarrowRegionT1) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(getProblem(case_20), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsNarrowRegionT2) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(getProblem(case_25), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT1) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
    SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    TestSolverDense(getProblem(case_21), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT2) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    TestSolverDense(getProblem(case_22), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT3) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    TestSolverDense(getProblem(case_23), NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
INSTANTIATE_TEST_CASE_P(SUITE_1, HessianParametrizedTest, ::testing::Values(1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0));

TEST(Solver, RedundantConstraintsT1) {
	QPBaseline baseline;
	baseline.xOpt = {{0.0, 2.0}};
	baseline.cost = 20.0;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_17, NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST(Solver, RedundantConstraintsT2) {
	QPBaseline baseline;
	baseline.xOpt = { {0.0, 2.0} };
	baseline.cost = 20.0;
    baseline.primalStatus = PrimalLoopExitStatus::ALL_PRIMAL_POSITIVE;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_24, NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST(Solver, FeasibleInitialPointT1) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.5, -0.5} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_1.c[0] * case_fsbl_init_pnt_1.c[0] + case_fsbl_init_pnt_1.c[1] * case_fsbl_init_pnt_1.c[1]);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_fsbl_init_pnt_1, NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST(Solver, FeasibleInitialPointT2) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.5, 0.5} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_2.c[0] * case_fsbl_init_pnt_2.c[0] + case_fsbl_init_pnt_2.c[1] * case_fsbl_init_pnt_2.c[1]);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_fsbl_init_pnt_2, NqpTestSettingsDefault, baseline, "testHessParam.txt");
}
TEST(Solver, FeasibleInitialPointT3) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.9999, 0.9999} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_3.c[0] * case_fsbl_init_pnt_3.c[0] + case_fsbl_init_pnt_3.c[1] * case_fsbl_init_pnt_3.c[1]);
    baseline.primalStatus = PrimalLoopExitStatus::DIDNT_STARTED;
    baseline.dualStatus = DualLoopExitStatus::ALL_DUAL_POSITIVE;
    TestSolverDense(case_fsbl_init_pnt_3, NqpTestSettingsDefault, baseline, "testHessParam.txt");
}

INSTANTIATE_TEST_CASE_P(SUITE_1, CompareRelHessParametrized, ::testing::Values(1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0));

#define QPTest QPTestCost
TEST_F(QPTest, CASE_1) {
	comparator.Compare(case_1, "case_1.txt");
}
TEST_F(QPTest, CASE_2) {
	comparator.Compare(case_2, "case_2.txt");
}
TEST_F(QPTest, CASE_3) {
	comparator.Compare(case_3, "case_3.txt");
}
TEST_F(QPTest, CASE_4) {
	comparator.Compare(case_4, "case_4.txt");
}
TEST_F(QPTest, CASE_5) {
	comparator.Compare(case_5, "case_5.txt");
}
TEST_F(QPTest, CASE_6) {
	comparator.Compare(case_6, "case_6.txt");
}
TEST_F(QPTest, CASE_7) {
	comparator.Compare(case_7, "case_7.txt");
}
TEST_F(QPTest, CASE_8) {
	comparator.Compare(case_8, "case_8.txt");
}
TEST_F(QPTest, CASE_9) {
	comparator.Compare(case_9, "case_9.txt");
}
TEST_F(QPTest, CASE_10) {
	comparator.Compare(case_10, "case_10.txt");
}
TEST_F(QPTest, CASE_11) {
	comparator.Compare(case_11, "case_11.txt");
}
TEST_F(QPTest, CASE_12) {
	comparator.Compare(case_12, "case_12.txt");
}
TEST_F(QPTest, CASE_13) {
	comparator.Compare(case_13, "case_13.txt");
}
TEST_F(QPTest, CASE_14) {
	comparator.Compare(case_14, "case_14.txt");
}
TEST_F(QPTest, CASE_15) {
	comparator.Compare(case_15, "case_15.txt");
}
TEST_F(QPTest, CASE_16) {
	comparator.Compare(case_16, "case_16.txt");
}
TEST_F(QPTest, S2F_0_9) {
	comparator.Compare(s2f_0_9, "s2f_0_9.txt");
}
TEST_F(QPTest, S2F_0_12) {
	comparator.Compare(s2f_0_12, "s2f_0_12.txt");
}
TEST_F(QPTest, C0_0_9) {
	comparator.Compare(c0_0_9, "c0_0_9.txt");
}
TEST_F(QPTest, C0_0_9_RELAXED1) {
    comparator.Compare(c0_0_9_relaxed1, "c0_0_9_rlx1.txt");
}
TEST_F(QPTest, C0_0_9_RELAXED2) {
    comparator.Compare(c0_0_9_relaxed2, "c0_0_9_rlx2.txt");
}
TEST_F(QPTest, C0_0_9_RELAXED3) {
    comparator.Compare(c0_0_9_relaxed3, "c0_0_9_rlx3.txt");
}
TEST_F(QPTest, C0_0_9_BOUNDS) {
    comparator.Compare(c0_0_9_bounds, "c0_0_9_bnds.txt");
}
TEST_F(QpTester, CASE_1) {
    Test(case_1, "case_1");
}
TEST_F(QpTester, CASE_2) {
    Test(case_2, "case_2");
}
TEST_F(QpTester, CASE_3) {
    Test(case_3, "case_3");
}
TEST_F(QpTester, CASE_4) {
    Test(case_4, "case_4");
}
TEST_F(QpTester, S2_0_9) {
    Test(s2f_0_9, "s2f_0_9");
}
TEST_F(QpTester, S2_0_12) {
    Test(s2f_0_12, "s2f_0_12");
}
TEST_F(QpTester, C0_0_9) {
    Test(c0_0_9, "c0_0_9");
}
// small
TEST_F(QpTesterMM, HS21) {
    Test("HS21");
}
TEST_F(QpTesterMM, HS35) {
    Test("HS35");
}
TEST_F(QpTesterMM, HS76) {
    Test("HS76");
}
TEST_F(QpTesterMM, HS268) {
    Test("HS268");
}
TEST_F(QpTesterMM, HS118) {
    Test("HS118");
}
TEST_F(QpTesterMM, QPTEST) {
    Test("QPTEST");
}
TEST_F(QpTesterMM, S268) {
    Test("S268");
}
TEST_F(QpTesterMM, ZECEVIC2) {
    Test("ZECEVIC2");
}
// medium
TEST_F(QpTesterMM, KSIP) {
    Test("KSIP");
}
TEST_F(QpTesterMM, QISRAEL) {
    Test("QISRAEL");
}
TEST_F(QpTesterMM, PRIMALC1) {
    Test("PRIMALC1");
}
TEST_F(QpTesterMM, PRIMALC2) {
    Test("PRIMALC2");
}
TEST_F(QpTesterMM, PRIMALC5) {
    Test("PRIMALC5");
}
TEST_F(QpTesterMM, PRIMALC8) {
    Test("PRIMALC8");
}
TEST_F(QpTesterMM, PRIMAL1) {
    Test("PRIMAL1");
}
//large
TEST_F(QpTesterMM, PRIMAL2) {
    Test("PRIMAL2");
}
TEST_F(QpTesterMM, PRIMAL3) {
    Test("PRIMAL3");
}
TEST_F(QpTesterMM, MOSARQP2) {
    Test("MOSARQP2");
}
// linEqC
// small
TEST_F(QpTesterMM, EQ_HS35MOD) {
    Test("HS35MOD", false);
}
TEST_F(QpTesterMM, EQ_TAME) {
    Test("TAME", false);
}
TEST_F(QpTesterMM, EQ_HS51) {
    Test("HS51", false);
}
TEST_F(QpTesterMM, EQ_HS52) {
    Test("HS52", false);
}
TEST_F(QpTesterMM, EQ_HS53) {
    Test("HS53", false);
}
TEST_F(QpTesterMM, EQ_GENHS28) {
    Test("GENHS28", false);
}
TEST_F(QpTesterMM, EQ_LOTSCHD) {
    Test("LOTSCHD", false);
}
TEST_F(QpTesterMM, EQ_DUALC1) {
    Test("DUALC1", false);
}
TEST_F(QpTesterMM, EQ_DUALC2) {
    Test("DUALC2", false);
}
TEST_F(QpTesterMM, EQ_QAFIRO) {
    Test("QAFIRO", false);
}
TEST_F(QpTesterMM, EQ_DUALC5) {
    Test("DUALC5", false);
}
TEST_F(QpTesterMM, EQ_DUALC8) {
    Test("DUALC8", false);
}
//medium
TEST_F(QpTesterMM, EQ_DUAL1) {
    Test("DUAL1", false);
}
TEST_F(QpTesterMM, EQ_DUAL2) {
    Test("DUAL2", false);
}
TEST_F(QpTesterMM, EQ_DUAL4) {
    Test("DUAL4", false);
}
TEST_F(QpTesterMM, EQ_QPCBLEND) {
    Test("QPCBLEND", false);
}
TEST_F(QpTesterMM, EQ_QSHARE2B) {
    Test("QSHARE2B", false);
}
TEST_F(QpTesterMM, EQ_CVXQP2_S) {
    Test("CVXQP2_S", false);
}
TEST_F(QpTesterMM, EQ_QADLITTL) {
    Test("QADLITTL", false);
}
TEST_F(QpTesterMM, EQ_DUAL3) {
    Test("DUAL3", false);
}
TEST_F(QpTesterMM, EQ_CVXQP1_S) {
    Test("CVXQP1_S", false);
}
TEST_F(QpTesterMM, EQ_CVXQP3_S) {
    Test("CVXQP3_S", false);
}
TEST_F(QpTesterMM, EQ_QSCAGR7) {
    Test("QSCAGR7", false);
}
TEST_F(QpTesterMM, EQ_DPKLO1) {
    Test("DPKLO1", false);
}
TEST_F(QpTesterMM, EQ_QPCBOEI2) {
    Test("QPCBOEI2", false);
}
TEST_F(QpTesterMM, EQ_QRECIPE) {
    Test("QRECIPE", false);
}
TEST_F(QpTesterMM, EQ_VALUES) {
    Test("VALUES", false);
}
//large
TEST_F(QpTesterMM, EQ_QSC205) {
    Test("QSC205", false);
}
TEST_F(QpTesterMM, EQ_QSHARE1B) {
    Test("QSHARE1B", false);
}
TEST_F(QpTesterMM, EQ_QBRANDY) {
    Test("QBRANDY", false);
}
TEST_F(QpTesterMM, EQ_QBEACONF) {
    Test("QBEACONF", false);
}
TEST_F(QpTesterMM, EQ_QE226) {
    Test("QE226", false);
}
TEST_F(QpTesterMM, EQ_QGROW7) {
    Test("QGROW7", false);
}
TEST_F(QpTesterMM, EQ_QBORE3D) {
    Test("QBORE3D", false);
}
TEST_F(QpTesterMM, EQ_QSCORPIO) {
    Test("QSCORPIO", false);
}
TEST_F(QpTesterMM, EQ_QCAPRI) {
    Test("QCAPRI", false);
}
TEST_F(QpTesterMM, EQ_QFORPLAN) {
    Test("QFORPLAN", false);
}
TEST_F(QpTesterMM, EQ_QPCBOEI1) {
    Test("QPCBOEI1", false);
}
TEST_F(QpTesterMM, EQ_QSCFXM1) {
    Test("QSCFXM1", false);
}
TEST_F(QpTesterMM, EQ_QBANDM) {
    Test("QBANDM", false);
}
TEST_F(QpTesterMM, EQ_QPCSTAIR) {
    Test("QPCSTAIR", false);
}
TEST_F(QpTesterMM, EQ_QSTAIR) {
    Test("QSTAIR", false);
}
TEST_F(QpTesterMM, EQ_QSCTAP1) {
    Test("QSCTAP1", false);
}
TEST_F(QpTesterMM, EQ_QSCAGR25) {
    Test("QSCAGR25", false);
}
// > 10 MB
TEST_F(QpTesterMM, EQ_QGROW15) {
    Test("QGROW15", false);
}
TEST_F(QpTesterMM, EQ_QSCSD1) {
    Test("QSCSD1", false);
}
TEST_F(QpTesterMM, EQ_GOULDQP3) {
    Test("GOULDQP3", false);
}
TEST_F(QpTesterMM, EQ_GOULDQP2) {
    Test("GOULDQP2", false);
}
TEST_F(QpTesterMM, EQ_QETAMACR) {
    Test("QETAMACR", false);
}
TEST_F(QpTesterMM, EQ_QFFFFF80) {
    Test("QFFFFF80", false);
}
TEST_F(QpTesterMM, EQ_QGROW22) {
    Test("QGROW22", false);
}
TEST_F(QpTesterMM, EQ_CVXQP2_M) {
    Test("CVXQP2_M", false);
}
TEST_F(QpTesterMM, EQ_QSCFXM2) {
    Test("QSCFXM2", false);
}
TEST_F(QpTesterMM, EQ_CVXQP1_M) {
    Test("CVXQP1_M", false);
}
TEST_F(QpTesterMM, EQ_CVXQP3_M) {
    Test("CVXQP3_M", false);
}










