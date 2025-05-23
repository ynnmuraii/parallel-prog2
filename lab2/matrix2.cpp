#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
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
   string filea = "matrixA.txt";
   string fileb = "matrixB.txt";
   string filec = "matrixC.txt";

   omp_set_num_threads(4);

   matrix a = read_matrix(filea);
   matrix b = read_matrix(fileb);

   auto start = chrono::high_resolution_clock::now();
   matrix c = multiply(a, b);
   auto end = chrono::high_resolution_clock::now();
   chrono::duration<double> diff = end - start;

   write_matrix(filec, c);

   cout << std::fixed << std::setprecision(6);
   cout << "execution time: " << diff.count() << " seconds" << endl;
   cout << "problem size: " << a.size() << "x" << a[0].size() << " * " << b.size() << "x" << b[0].size() << endl;

   return 0;
}