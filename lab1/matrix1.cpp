#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>

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

Matrix multiply(const Matrix& a, const Matrix& b) {
    int n = a.size(), m = a[0].size(), p = b[0].size();
    Matrix c(n, vector<double>(p, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k)
            for (int j = 0; j < p; ++j)
                c[i][j] += a[i][k] * b[k][j];
    return c;
}

int main() {
    string fileA = "matrixA.txt";
    string fileB = "matrixB.txt";
    string fileC = "matrixC.txt";

    Matrix a = read_matrix(fileA);
    Matrix b = read_matrix(fileB);

    auto start = chrono::high_resolution_clock::now();
    Matrix c = multiply(a, b);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    write_matrix(fileC, c);

    cout << std::fixed << std::setprecision(6);
    cout << "Execution time: " << diff.count() << " seconds" << endl;
    cout << "Problem size: " << a.size() << "x" << a[0].size() << " * " << b.size() << "x" << b[0].size() << endl;

    return 0;
}