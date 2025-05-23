#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <omp.h> // openmp

using namespace std;

typedef vector<vector<double>> matrix;

matrix read_matrix(const string& filename) {
    ifstream fin(filename);
    int rows, cols;
    fin >> rows >> cols;
    matrix m(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            fin >> m[i][j];
    return m;
}

void write_matrix(const string& filename, const matrix& m) {
   ofstream fout(filename);
   int rows = m.size(), cols = m[0].size();
   fout << rows << " " << cols << endl;
   for (int i = 0; i < rows; ++i) {
       for (int j = 0; j < cols; ++j)
           fout << m[i][j] << " ";
       fout << endl;
   }
}

matrix multiply(const matrix& a, const matrix& b) {
   int n = a.size(), m = a[0].size(), p = b[0].size();
   matrix c(n, vector<double>(p, 0.0));
   #pragma omp parallel for collapse(2) schedule(static)
   for (int i = 0; i < n; ++i)
       for (int j = 0; j < p; ++j) {
           double sum = 0.0;
           for (int k = 0; k < m; ++k)
               sum += a[i][k] * b[k][j];
           c[i][j] = sum;
       }
   return c;
}

int main() {
    vector<int> sizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
    ofstream res("../../../lab2/results_lab2.txt");
    vector<int> thread_counts = {2, 4, 8};
    for (int threads : thread_counts) {
        for (int sz : sizes) {
            string fileA = "../../../matrix/matrixA_" + to_string(sz) + ".txt";
            string fileB = "../../../matrix/matrixB_" + to_string(sz) + ".txt";
            string fileC = "../../../matrix/matrixC_" + to_string(sz) + ".txt";

            omp_set_num_threads(threads);

            matrix a = read_matrix(fileA);
            matrix b = read_matrix(fileB);

            auto start = chrono::high_resolution_clock::now();
            matrix c = multiply(a, b);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, std::milli> diff = end - start;

            write_matrix(fileC, c);

            res << "Threads: " << threads << " | Matrix size: " << a.size() << "x" << a[0].size() << " * " << b.size() << "x" << b[0].size()
                << " | Execution time: " << diff.count() << " ms" << endl;

            cout << std::fixed << std::setprecision(3);
            cout << "Threads: " << threads << " | Execution time: " << diff.count() << " ms" << endl;
            cout << "matrix size: " << a.size() << "x" << a[0].size() << " * " << b.size() << "x" << b[0].size() << endl;
        }
    }
    return 0;
}