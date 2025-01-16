
#ifndef NNLS_QP_SOLVER_UTILS_H
#define NNLS_QP_SOLVER_UTILS_H
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include "types.h"
namespace QP_NNLS {

	void ComputeCholFactor(const matrix_t& M, matrix_t& cholF) ; // M = cholF * cholF_T

	bool ComputeCholFactorT(const matrix_t& M, matrix_t& cholF, CholetskyOutput& output); // M = cholF_T * cholF

	int ComputeCholFactorTFullPivoting(matrix_t& M, matrix_t& cholF, std::vector<int>& permut); // P_T * M * P = cholF_T * cholF

	void Mult(const matrix_t& M1, const matrix_t& M2, matrix_t& mult); // M1 * M2

	void MultTransp(const matrix_t& M, const std::vector<double>& v, std::vector<double>& MTv); // MTv = M_T * v

	void MultTransp(const matrix_t& M, const std::vector<double>& v, const std::vector<int>& activesetIndices, std::vector<double>& MTv); // M_T * v on active set

    void MultTransp(const matrix_t& M, const std::vector<double>& v, const std::set<unsg_t>& activesetIndices, std::vector<double>& MTv); // M_T * v on active set

	void M1M2T(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT); // MMT = M1 * M2_T

	void M2M1T(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT); // MMT = M2 * M1_T

	void M1TM2(const matrix_t& M1, const matrix_t& M2, matrix_t& MMT); // MMT = M1_T * M2

	void swapColumns(matrix_t& M, int r1, int r2);

	void InvertTriangle(const matrix_t& M, matrix_t& Minv);// M^-1 M -low triangular matrix

	void Mult(const matrix_t& M, const std::vector<double>& v, std::vector<double>& Mv); // Mv = M * v

	void VSum(const std::vector<double>& v1, const std::vector<double>& v2, std::vector<double>& sum); // sum = v1 + v2

	void VAdd(std::vector<double>& v1, const std::vector<double>& v2); // v1 += v2

	double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2); // <v1,v2>

	double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2,  const std::vector<int>& activeSetIndices); // <v1,v2> for active set indices 

    double DotProduct(const std::vector<double>& v1, const std::vector<double>& v2,  const std::set<unsg_t>& activesetIndices); // <v1,v2> for active set indices

	void InvertByGauss(const matrix_t& M, matrix_t& Minv); // invert matrix M using Gauss Elimination with pivoting, M_inv must be filled with zeros in advance

	void InvertLTrByGauss(const matrix_t& M, matrix_t& Minv); // invert low triangular matrix M using Gauss Elimination with pivoting, M_inv must be filled with zeros in advance

	void InvertEigen(const matrix_t& M, matrix_t& Inv); // invert matrix M using Eigen library; inverse() method

	void GetIdentityMatrix(int size, matrix_t& M); // set M as identity matrix 

	double bTAb(const std::vector<double>& b, const matrix_t& A); // b_T * A * b

	void RRF(matrix_t& M); // Transform M to reduced row echelone form

	void RRFB(matrix_t& matrix, std::vector<double>& b); // Transform  M to reduced row echelone form and b apply all the changes to vector b 

	void RCFB(matrix_t& A, std::vector<double>& b); // Transform M to reduced column echelone form

	void MultStrictLowTriangular(const matrix_t& M1, const matrix_t& M2, matrix_t& M1M2, int m1Bg = 1, int m2Bg = 1); // M1M2 = M1 * M2 , M1 and M2 -are strict low triangular

	void InvertTriangularFactorization(const matrix_t& M, matrix_t& Minv); // M-1 Minv, invert low triangular matrix using factorization method

	void PermuteColumns(matrix_t& A, const std::vector<int>& pmt);  // swaps columns of matrix A: A[i] <-> A[pmt[i]]

    void PTV(std::vector<double>& v, const std::vector<int>& pmt); // v -> P_T * v

    void InvertHermit(const matrix_t& Chol, matrix_t& Inv); // invert hemitian matrix M using it's Choletsky decomposition M = L * L_T

    void InvertCholetsky(const matrix_t& Chol, matrix_t& Inv); // invert hemitian matrix M using it's Choletsky decomposition M = L * L_T

	matrix_t& operator-(matrix_t& M); // M -> -M

	static inline bool isSame(double cand, double val, double tol = 1.0e-16) {
		assert(tol > 0.0);
		const double diff = cand - val;
		return ((diff >= -tol) && (diff <= tol));
	}

    class LDLT
    {
    public:
        LDLT() = delete;
        LDLT(const matrix_t& M, const std::vector<double>& S);
        virtual ~LDLT() = default;
        void Compute(const std::set<unsigned int>& activeColumns);
        const matrix_t& GetL() { return L;}
        const std::vector<double>& GetD() { return D;}
    private:
        const double dTol = 1.0e-16;
        double d;
        const std::size_t maxSize;
        const std::size_t nX;
        std::size_t curIndex;
        std::size_t actSize;
        const matrix_t& M;
        const std::vector<double>& S;
        matrix_t L;
        std::vector<double> D;
        std::vector<double> norms2;
    };

    class LDL
    {
    public:
        // L*D*LT = A*AT
        LDL() = default;
        virtual ~LDL() = default;
        void Set(const matrix_t& A);
        void Compute();
        void Add(const std::vector<double>& row);
        void Remove(int i);
        const matrix_t& GetL();
        const std::vector<double>& GetD();
    protected:
        int dimR = 0;
        int dimC = 0;
        int curIndex = 0;
        double d = 0.0;
        matrix_t L;
        std::vector<double> D;
        matrix_t A;
        std::vector<double> l;
        void compute_l();
        void compute_d();
        void update_L();
        void update_D();
        void solveLDb(const std::vector<double>& b, std::vector<double>& l);
        double getARowNormSquared(int row) const;
        void update_L_remove(int iRow, const matrix_t& Ltil);
        std::vector<int> activeRows;
    };

    class MMTbSolver
    {
    public:
        MMTbSolver() = default;
        virtual ~MMTbSolver() = default;
        int Solve(const matrix_t& M, const std::vector<double>& b);
        int nDZero();
        const std::vector<double>& GetSolution();
    protected:
        void SolveForward(const matrix_t& L, const std::vector<double>& b);
        void SolveBackward(const std::vector<double>& D, const matrix_t& L);
        void GetMMTKernel(const std::vector<int>& dzeroIndices, const matrix_t& L,std::vector<double>& ker);
        std::vector<double> solution;
        std::vector<double> forward;
        std::vector<double> backward;
        const double zeroTol = 1.0e-16;
        int ndzero = 0;
    };
}
#endif
