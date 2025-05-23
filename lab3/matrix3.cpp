#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <mpi.h>

using namespace std;

typedef vector<vector<double>> Matrix;

Matrix read_matrix(const string& filename) {
    ifstream fin(filename);
    int rows, cols;
    fin >> rows >> cols;
    Matrix m(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            fin >> m[i][j];
    return m;
}

void write_matrix(const string& filename, const Matrix& m) {
    ofstream fout(filename);
    int rows = m.size(), cols = m[0].size();
    fout << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            fout << m[i][j] << " ";
        fout << endl;
    }
}

void local_multiply(const vector<double>& localA, const vector<double>& B, vector<double>& localC,
                    int local_n, int m, int p) {
    for (int i = 0; i < local_n; ++i)
        for (int k = 0; k < m; ++k)
            for (int j = 0; j < p; ++j)
                localC[i * p + j] += localA[i * m + k] * B[k * p + j];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    ofstream res;
    string res_file;
    if (rank == 0) {
        res_file = "../../../lab3/results_lab3_" + to_string(size) + ".txt";
        res.open(res_file, ios::app);
    }

    for (int sz : sizes) {
        string fileA = "../../../matrix/matrixA_" + to_string(sz) + ".txt";
        string fileB = "../../../matrix/matrixB_" + to_string(sz) + ".txt";
        string fileC = "../../../matrix/matrixC_" + to_string(sz) + ".txt";

        int n, m, p;
        vector<double> flatA, flatB;

        if (rank == 0) {
            Matrix a = read_matrix(fileA);
            Matrix b = read_matrix(fileB);
            n = a.size(); m = a[0].size(); p = b[0].size();
            flatA.resize(n * m);
            flatB.resize(m * p);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    flatA[i * m + j] = a[i][j];
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < p; ++j)
                    flatB[i * p + j] = b[i][j];
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0)
            flatB.resize(m * p);
        MPI_Bcast(flatB.data(), m * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int rows_per_proc = n / size;
        int extra = n % size;
        int local_n = rows_per_proc + (rank < extra ? 1 : 0);
        int offset = rank * rows_per_proc + min(rank, extra);

        vector<int> sendcounts(size), displs(size);
        if (rank == 0) {
            for (int r = 0, offs = 0; r < size; ++r) {
                int r_n = rows_per_proc + (r < extra ? 1 : 0);
                sendcounts[r] = r_n * m;
                displs[r] = offs;
                offs += r_n * m;
            }
        }

        vector<double> localA(local_n * m);
        MPI_Scatterv(rank == 0 ? flatA.data() : nullptr,
                     rank == 0 ? sendcounts.data() : nullptr,
                     rank == 0 ? displs.data() : nullptr,
                     MPI_DOUBLE,
                     localA.data(), local_n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        vector<double> localC(local_n * p, 0.0);
        MPI_Barrier(MPI_COMM_WORLD); 
        auto start = chrono::high_resolution_clock::now();
        local_multiply(localA, flatB, localC, local_n, m, p);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> diff = end - start;

        vector<int> recvcounts(size), recvdispls(size);
        if (rank == 0) {
            for (int r = 0, offs = 0; r < size; ++r) {
                int r_n = rows_per_proc + (r < extra ? 1 : 0);
                recvcounts[r] = r_n * p;
                recvdispls[r] = offs;
                offs += r_n * p;
            }
        }
        vector<double> flatC;
        if (rank == 0)
            flatC.resize(n * p);

        MPI_Gatherv(localC.data(), local_n * p, MPI_DOUBLE,
                    rank == 0 ? flatC.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? recvdispls.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            Matrix c(n, vector<double>(p));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < p; ++j)
                    c[i][j] = flatC[i * p + j];
            write_matrix(fileC, c);
            res << "Cores: " << size << " | Matrix size: " << n << "x" << m << " * " << m << "x" << p
                << " | Execution time: " << diff.count() << " ms" << endl;
            cout << std::fixed << std::setprecision(3);
            cout << "Cores: " << size << " | Execution time: " << diff.count() << " ms" << endl;
            cout << "Matrix size: " << n << "x" << m << " * " << m << "x" << p << endl;
        }
    }

    if (rank == 0) res.close();
    MPI_Finalize();
    return 0;
}
