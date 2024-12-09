#include "utils.h"
#include "test_utils.h"
#include "test_data.h"
#include "NNLSQPSolver.h"
#include <algorithm>

using namespace QP_NNLS;
using namespace QP_NNLS_TEST_DATA;
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
#ifndef NEW_INTERFACE
	TestSolver(case_2, NqpTestSettingsDefault, baseline);
#else
    TestSolverDense(static_cast<DenseQPProblem>(case_2), NqpTestSettingsDefaultNewInterface, baseline, "test1.txt");
#endif
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest2) {
	QPBaseline baseline;
	baseline.xOpt = {{0.5, 1.5}};
	baseline.cost = 1.25;
	TestSolver(case_3, NqpTestSettingsDefault, baseline);
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest3) {
	QPBaseline baseline;
	baseline.xOpt = {{0.0, 1.0}};
	baseline.cost = 0.5;
	TestSolver(case_4, NqpTestSettingsDefault, baseline);
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest4) {
	QPBaseline baseline;
	baseline.xOpt = { {1.5, 2.5} };
	baseline.cost = 4.25;
	TestSolver(case_5, NqpTestSettingsDefault, baseline);
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest5) {
	QPBaseline baseline;
	baseline.xOpt = {{0, 1.0}};
	baseline.cost = 0.5;
	TestSolver(case_6, NqpTestSettingsDefault, baseline);
}
TEST(Solver, SolutionOnConstraintsIdentityHessTest6) {
	QPBaseline baseline;
	baseline.xOpt = { {-0.5, 1.5} };
	baseline.cost = 1.25;
	TestSolver(case_7, NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, SolutionOnConstraintsDiagHessSameValues) {
	QPBaseline baseline;
	baseline.xOpt = {{-0.5, 1.5}};
	baseline.cost = 1.25 * GetScaleFactor();
	SetModification(HessianParametrizedTest::Modification::DIAG_SCALING);
	TestSolver(getProblem(case_7), NqpTestSettingsDefault, baseline);
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
	TestSolver(getProblem(case_18), NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsT2) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	TestSolver(getProblem(case_19), NqpTestSettingsDefault, baseline);
}
TEST_P(CompareRelHessParametrized, Solver_SolNotOnConstraintsParametrizedT1) {
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	UserSettings settings = NqpTestSettingsDefault;
	settings.cholPvtStrategy = CholPivotingStrategy::FULL;
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
	TestSolver(getProblem(case_20), NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, Solver_SolNotOnConstraintsNarrowRegionT2) {
	QPBaseline baseline;
	baseline.xOpt = {{ 0.0, 0.0}};
	baseline.cost = 0.0;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	TestSolver(getProblem(case_25), NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT1) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
    SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	TestSolver(getProblem(case_21), NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT2) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	TestSolver(getProblem(case_22), NqpTestSettingsDefault, baseline);
}
TEST_P(HessianParametrizedTest, Solver_InfeasibleProblemT3) {
	QPBaseline baseline;
	baseline.dualStatus = DualLoopExitStatus::INFEASIBILITY;
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	TestSolver(getProblem(case_23), NqpTestSettingsDefault, baseline);
}
INSTANTIATE_TEST_CASE_P(SUITE_1, HessianParametrizedTest, ::testing::Values(1.0)); //, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0));

TEST(Solver, RedundantConstraintsT1) {
	QPBaseline baseline;
	baseline.xOpt = {{0.0, 2.0}};
	baseline.cost = 20.0;
    UserSettings settings = NqpTestSettingsDefault;
    settings.logLevel = 3;
    TestSolver(case_17, settings, baseline);
}
TEST(Solver, RedundantConstraintsT2) {
	QPBaseline baseline;
	baseline.xOpt = { {0.0, 2.0} };
	baseline.cost = 20.0;
	TestSolver(case_24, NqpTestSettingsDefault, baseline);
}
TEST(Solver, FeasibleInitialPointT1) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.5, -0.5} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_1.c[0] * case_fsbl_init_pnt_1.c[0] + case_fsbl_init_pnt_1.c[1] * case_fsbl_init_pnt_1.c[1]);
	TestSolver(case_fsbl_init_pnt_1, NqpTestSettingsDefault, baseline);
}
TEST(Solver, FeasibleInitialPointT2) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.5, 0.5} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_2.c[0] * case_fsbl_init_pnt_2.c[0] + case_fsbl_init_pnt_2.c[1] * case_fsbl_init_pnt_2.c[1]);
	TestSolver(case_fsbl_init_pnt_2, NqpTestSettingsDefault, baseline);
}
TEST(Solver, FeasibleInitialPointT3) {
	QPBaseline baseline;
	baseline.xOpt =  { {-0.9999, 0.9999} };
	baseline.cost = -0.5 * (case_fsbl_init_pnt_3.c[0] * case_fsbl_init_pnt_3.c[0] + case_fsbl_init_pnt_3.c[1] * case_fsbl_init_pnt_3.c[1]);
	TestSolver(case_fsbl_init_pnt_3, NqpTestSettingsDefault, baseline);
}
TEST_P(CompareRelHessParametrized, Solver_CompareQLDCholFullPivotingParametrizedT1) {
	SetModification(HessianParametrizedTest::Modification::STRATEGY_1);
	UserSettings settings = NqpTestSettingsDefault;
	settings.cholPvtStrategy = CholPivotingStrategy::FULL;
	this->settings.uSettings = settings;
	QPProblem problem;
	problem.c = { 0.0, 0.0 };
	problem.A = { { 0.0, -1.0 } };
	problem.b = { -1.0 };
	comparator.Compare(getProblem(problem), "case_full_pivoting_prm1_" + std::to_string(static_cast<int>(GetParam())) + ".txt");
}
INSTANTIATE_TEST_CASE_P(SUITE_1, CompareRelHessParametrized, ::testing::Values(1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0));

TEST_F(QPTestRelative, Solver_CompareQLDCholFullPivotingT1) {
	settings.uSettings.cholPvtStrategy = CholPivotingStrategy::FULL;
	comparator.Compare(case_chol_full_pvt_1, "chol_full_pivoting_t1.txt");
}
TEST_F(QPTestRelative, Solver_CompareQLDCholFullPivotingT2) {
	settings.uSettings.cholPvtStrategy = CholPivotingStrategy::FULL;
	comparator.Compare(case_chol_full_pvt_2, "chol_full_pivoting_t2.txt");
}
TEST_F(QPTestRelative, Solver_CompareQLDCholFullPivotingT3) {
	settings.uSettings.cholPvtStrategy = CholPivotingStrategy::FULL;
	QPProblem problem = case_chol_full_pvt_3;
	//add corrections to hessian to activate pivoting
	double corr = 1.0;
	const double corrIncr = 100.0;
	GetIdentityMatrix(problem.c.size(), problem.H);
    for (int i = problem.H.size() - 1; i >= 0; --i) {
		problem.H[i][i] = corr;
		corr += corrIncr;
	}
	comparator.Compare(problem, "chol_full_pivoting_t3.txt");
}
TEST_F(QPTestRelative, CASE_1) {
	comparator.Compare(case_1, "case_1.txt");
}
TEST_F(QPTestRelative, CASE_2) {
	comparator.Compare(case_2, "case_2.txt");
}
TEST_F(QPTestRelative, CASE_3) {
	comparator.Compare(case_3, "case_3.txt");
}
TEST_F(QPTestRelative, CASE_4) {
	comparator.Compare(case_4, "case_4.txt");
}
TEST_F(QPTestRelative, CASE_5) {
	comparator.Compare(case_5, "case_5.txt");
}
TEST_F(QPTestRelative, CASE_6) {
	comparator.Compare(case_6, "case_6.txt");
}
TEST_F(QPTestRelative, CASE_7) {
	comparator.Compare(case_7, "case_7.txt");
}
TEST_F(QPTestRelative, CASE_8) {
	comparator.Compare(case_8, "case_8.txt");
}
TEST_F(QPTestRelative, CASE_9) {
	comparator.Compare(case_9, "case_9.txt");
}
TEST_F(QPTestRelative, CASE_10) {
	comparator.Compare(case_10, "case_10.txt");
}
TEST_F(QPTestRelative, CASE_11) {
	comparator.Compare(case_11, "case_11.txt");
}
TEST_F(QPTestRelative, CASE_12) {
	comparator.Compare(case_12, "case_12.txt");
}
TEST_F(QPTestRelative, CASE_13) {
	comparator.Compare(case_13, "case_13.txt");
}
TEST_F(QPTestRelative, CASE_14) {
	comparator.Compare(case_14, "case_14.txt");
}
TEST_F(QPTestRelative, CASE_15) {
	comparator.Compare(case_15, "case_15.txt");
}
TEST_F(QPTestRelative, CASE_16) {
	comparator.Compare(case_16, "case_16.txt");
}
TEST_F(QPTestRelative, S2F_0_9) {
	comparator.Compare(s2f_0_9, "s2f_0_9.txt");
}
TEST_F(QPTestRelative, S2F_0_12) {
	comparator.Compare(s2f_0_12, "s2f_0_12.txt");
}
TEST_F(QPTestRelative, C0_0_9) {
	comparator.Compare(c0_0_9, "c0_0_9.txt");
}
TEST_F(QPTestCost, CASE_1) {
	comparator.Compare(case_1, "case_1.txt");
}
TEST_F(QPTestCost, CASE_2) {
	comparator.Compare(case_1, "case_2.txt");
}
TEST_F(QPTestCost, S2F_0_9) {
	comparator.Compare(s2f_0_9, "s2f_0_9.txt");
}
TEST_F(QPTestCost, S2F_0_12) {
	comparator.Compare(s2f_0_12, "s2f_0_12.txt");
}
TEST_F(QPTestCost, C0_0_9) {
	comparator.Compare(c0_0_9, "c0_0_9.txt");
}







