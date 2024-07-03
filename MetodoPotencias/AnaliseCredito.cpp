#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

// Função para ler matriz do arquivo
vector<vector<double>> readMatrix(const string& filename, int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo." << endl;
        exit(1);
    }

    vector<vector<double>> matrix(rows, vector<double>(cols));
    string line;
    for (int i = 0; i < rows && getline(file, line); ++i) {
        istringstream iss(line);
        for (int j = 0; j < cols && iss >> matrix[i][j]; ++j);
    }

    file.close();
    return matrix;
}

// Função para imprimir uma matriz
void printMatrix(const vector<vector<double>>& matrix) {
    for (const vector<double>& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Função para multiplicar duas matrizes
vector<vector<double>> multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    vector<vector<double>> result(rowsA, vector<double>(colsB, 0.0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Função para transpor uma matriz
vector<vector<double>> transposeMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> transposed(cols, vector<double>(rows, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Função para resolver um sistema de equações lineares Ax = b usando eliminação de Gauss
vector<double> solveLinearSystem(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<vector<double>> augmentedMatrix(n, vector<double>(n + 1, 0.0));

    // Construir a matriz aumentada [A|b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmentedMatrix[i][j] = A[i][j];
        }
        augmentedMatrix[i][n] = b[i];
    }

    // Aplicar eliminação de Gauss
    for (int i = 0; i < n; ++i) {
        // Pivotar
        for (int k = i + 1; k < n; ++k) {
            if (fabs(augmentedMatrix[i][i]) < fabs(augmentedMatrix[k][i])) {
                swap(augmentedMatrix[i], augmentedMatrix[k]);
            }
        }
        // Eliminar
        for (int k = i + 1; k < n; ++k) {
            double t = augmentedMatrix[k][i] / augmentedMatrix[i][i];
            for (int j = 0; j <= n; ++j) {
                augmentedMatrix[k][j] -= t * augmentedMatrix[i][j];
            }
        }
    }

    // Resolver sistema triangular superior
    vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = augmentedMatrix[i][n] / augmentedMatrix[i][i];
        for (int j = i - 1; j >= 0; --j) {
            augmentedMatrix[j][n] -= augmentedMatrix[j][i] * x[i];
        }
    }

    return x;
}

// Função para calcular a matriz de fatores F usando o Método dos Mínimos Quadrados
vector<vector<double>> calculateF(const vector<vector<double>>& A, const vector<vector<double>>& Y) {
    int n = Y.size(); // Número de amostras (25)
    int m = A[0].size(); // Número de colunas em A
    int p = A.size(); // Número de linhas em A (7)

    // Inicializar a matriz F
    vector<vector<double>> F(n, vector<double>(m, 0.0));

    // Calcular A^T * A
    vector<vector<double>> A_transposed = transposeMatrix(A);
    vector<vector<double>> A_transposed_A = multiplyMatrices(A_transposed, A);

    // Para cada i de 1 a 25
    for (int i = 0; i < n; ++i) {
        vector<double> Y_col(p, 0.0);
        for (int j = 0; j < p; ++j) {
            Y_col[j] = Y[i][j];
        }

        // Calcular A^T * Y_col
        vector<double> A_transposed_Y(m, 0.0);
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                A_transposed_Y[j] += A_transposed[j][k] * Y_col[k];
            }
        }

        // Resolver o sistema de equações pelo método dos mínimos quadrados
        vector<double> F_i = solveLinearSystem(A_transposed_A, A_transposed_Y);

        // Armazenar os fatores calculados na matriz F
        for (int k = 0; k < m; ++k) {
            F[i][k] = F_i[k];
        }
    }

    return F;
}

int main() {
    // Ler a matriz de correlação do arquivo correlacao.txt
    vector<vector<double>> correlacao = readMatrix("correlacao.txt", 7, 7);
    cout << "Matriz de Correlacao:\n";
    printMatrix(correlacao);

    // Dados de D e V (obtidos do trabalho 1)
    vector<vector<double>> D = {
        {3.34482, 0.0},
        {0.0, 2.16093}
    };

    vector<vector<double>> V = {
        {0.477844, -0.21913},
        {0.47794, -0.000873671},
        {0.394721, -0.248027},
        {0.472262, -0.0243972},
        {-0.0660366, 0.641765},
        {0.343833, 0.414636},
        {0.201561, 0.553225}
    };

    // Calcular a matriz A = V * sqrt(D)
    vector<vector<double>> A(7, vector<double>(2, 0.0));
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            A[i][j] = V[i][j] * sqrt(D[j][j]);
        }
    }

    cout << "\nMatriz A = V * sqrt(D):\n";
    printMatrix(A);

    // Ler a matriz 25x7 de dados do arquivo dados.txt
    vector<vector<double>> Y = readMatrix("dados.txt", 25, 7);
    cout << "\nMatriz de Dados (Y):\n";
    printMatrix(Y);

    // Calcular as médias das colunas de Y
    vector<double> columnMeans(7, 0.0);
    for (size_t j = 0; j < 7; ++j) {
        for (size_t i = 0; i < 25; ++i) {
            columnMeans[j] += Y[i][j];
        }
        columnMeans[j] /= 25;
    }

    cout << "\nMedias das colunas de Y:\n";
    for (double mean : columnMeans) {
        cout << mean << " ";
    }
    cout << endl;

    // Calcular os vetores Yj subtraindo cada entrada do vetor Y por sua média
    vector<vector<double>> Y_centered(25, vector<double>(7, 0.0));
    for (size_t j = 0; j < 7; ++j) {
        for (size_t i = 0; i < 25; ++i) {
            Y_centered[i][j] = Y[i][j] - columnMeans[j];
        }
    }

    cout << "\nVetores Yj centrados:\n";
    printMatrix(Y_centered);

    // Calcular a matriz de fatores F
    vector<vector<double>> F = calculateF(A, Y_centered);

    // Imprimir a matriz F
    cout << "\nMatriz de Fatores F:\n";
    printMatrix(F);
    cout << "\n";

    // Nomes das características
    vector<string> characteristics = {
        "Idade",
        "Emprego", 
        "Endereco",
        "Renda",
        "Divida",
        "Divida de cartao de credito",
        "Outras Dividas"
    };

    // (d) Determinar as características mais explicadas por cada fator
    for (int j = 0; j < 2; ++j) { // Iterar sobre os fatores
        double max_val = 0.0;
        int max_idx = 0;
        for (int i = 0; i < 7; ++i) { // Iterar sobre as características
            if (fabs(A[i][j]) > max_val) {
                max_val = fabs(A[i][j]);
                max_idx = i;
            }
        }
        cout << "Fator " << j + 1 << " mais explica a caracteristica " << characteristics[max_idx] << " com valor " << A[max_idx][j] << endl;
    }

    // (e) Determinar os três melhores clientes para oferecer crédito
    vector<pair<int, double>> client_scores;
    for (int i = 0; i < 25; ++i) {
        double score = 0.0;
        for (int j = 0; j < 2; ++j) {
            score += F[i][j];
        }
        client_scores.push_back(make_pair(i, score));
    }

    // Ordenar os clientes pelos scores em ordem decrescente
    sort(client_scores.begin(), client_scores.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
        return a.second > b.second;
        });

    // Imprimir os três melhores clientes
    cout << "\nTres melhores clientes para oferecer credito:\n";
    for (int i = 0; i < 3; ++i) {
        cout << "Cliente " << client_scores[i].first + 1 << " com score " << client_scores[i].second << endl;
    }

    return 0;
}
