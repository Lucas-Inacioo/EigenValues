#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <utility>

using namespace std;

// Método para ler matrix do arquivo corr.txt
vector<vector<double>> readMatrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo." << endl;
        exit(1);
    }

    vector<vector<double>> matrix;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}

// Método para multiplicar matriz por um vetor
vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {
    if (matrix.empty() || matrix[0].size() != vec.size()) {
        cerr << "Dimensões incompatíveis para multiplicação." << endl;
        exit(1);
    }

    vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    return result;
}

// Método para definir o vetor inicial, por padrão, um vetor de 'uns'
vector<double> defineInitialVector(size_t size) {
    vector<double> vec(size, 1.0);
    return vec;
}

// Método para calcular a norma de um vetor
double calculateVectorNorm(const vector<double>& vec) {
    double norm = 0.0;
    for (double val : vec) {
        // Soma os quadrados de cada valor do vetor
        norm += val * val;
    }
    // Raíz da soma dos quadrados
    return sqrt(norm);
}

// Método para resolver um sistema utilizando o método de Gauss (somente escalonamento)
void gauss(vector<vector<double>>& matrix, vector<double>& bValues) {
    int size = matrix.size();

    // Método para corrigir imprecisão float, arredodando valores próximos de zero
    auto roundToZero = [](double value) -> double {
        const double epsilon = 1e-10;
        return (fabs(value) < epsilon) ? 0.0 : value;
        };

    for (int actualLine = 0; actualLine < size - 1; actualLine++) {
        for (int line = actualLine + 1; line < size; line++) {
            double actualM = matrix[line][actualLine] / matrix[actualLine][actualLine];
            for (int column = 0; column < size; column++) {
                matrix[line][column] -= actualM * matrix[actualLine][column];
                matrix[line][column] = roundToZero(matrix[line][column]);
            }
            bValues[line] -= actualM * bValues[actualLine];
            bValues[line] = roundToZero(bValues[line]);
        }
    }
}

// Método para realizar substituição regressiva na matriz escalonada anteriormente
vector<double> backSubstitution(const vector<vector<double>>& matrix, const vector<double>& bValues) {
    int size = matrix.size();
    vector<double> solution(size, 0.0);

    for (int actualEquation = size - 1; int(actualEquation) >= 0; --actualEquation) {
        solution[actualEquation] = bValues[actualEquation];
        for (int variable = actualEquation + 1; variable < size; ++variable) {
            solution[actualEquation] -= matrix[actualEquation][variable] * solution[variable];
        }
        solution[actualEquation] /= matrix[actualEquation][actualEquation];
    }

    return solution;
}

// Método para unir os métodos de escalonamento e resolução do sistema linear, de forma a simplificar o código
vector<double> solveSystem(const vector<vector<double>>& A, const vector<double>& b) {
    vector<vector<double>> matrix = A;
    vector<double> bValues = b;
    gauss(matrix, bValues);
    return backSubstitution(matrix, bValues);
}

// Método auxiliar para printar uma matriz
void printMatrix(const vector<vector<double>>& matrix) {
    for (const vector<double>& linha : matrix) {
        for (double elemento : linha) {
            cout << elemento << " ";
        }
        cout << endl;
    }
}

// Método das potências
pair<double, vector<double>> powerMethod(const vector<vector<double>>& matrix, vector<double> initialVector, double tolerance, int maxIterations) {
    vector<double> y = initialVector;
    double lambda_old = 0.0;
    double lambda_new = 0.0;

    for (int i = 0; i < maxIterations; ++i) {
        y = multiplyMatrixVector(matrix, y);
        double norm = calculateVectorNorm(y);
        for (double& val : y) {
            val /= norm;
        }
        lambda_new = norm;
        if (fabs(lambda_new - lambda_old) < tolerance) {
            break;
        }
        lambda_old = lambda_new;
    }

    return make_pair(lambda_new, y);
}

// Método das potências inverso com deslocamento
pair<double, vector<double>> inversePowerMethodWithShift(const vector<vector<double>>& matrix, vector<double> initialVector, double shift, double tolerance, int maxIterations) {
    int n = matrix.size();
    vector<vector<double>> shiftedMatrix = matrix;

    for (int i = 0; i < n; ++i) {
        shiftedMatrix[i][i] -= shift;
    }

    vector<double> y = initialVector;
    double lambda_old = 0.0;
    double lambda_new = 0.0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        y = solveSystem(shiftedMatrix, y);

        double norm = calculateVectorNorm(y);
        for (double& val : y) {
            val /= norm;
        }
        lambda_new = 1.0 / norm + shift;

        if (fabs(lambda_new - lambda_old) < tolerance) {
            break;
        }
        lambda_old = lambda_new;
    }

    return make_pair(lambda_new, y);
}

// Método para checar se dois valores float são iguais (dado uma tolerância, padrão de 0.0001)
bool is_equal(double a, double b, double tol = 1e-4) {
    return fabs(a - b) <= tol;
}

// Método para printar autovalores com frequência maior que 1
void printEigenvaluesWithFrequency(const map<double, pair<int, vector<double>>>& eigen_map) {
    cout << "Autovalores com frequencia maior que 1:" << endl;
    for (const auto& entry : eigen_map) {
        if (entry.second.first > 1) {
            cout << "Autovalor: " << entry.first << " - Frequencia: " << entry.second.first << endl;
        }
    }
}

// Função principal do programa
int main() {
    // Ler e printar matriz do arquivo .txt
    vector<vector<double>> matrix = readMatrix("correlacao.txt");
    cout << "Matriz inicial obtida do arquivo:" << endl;
    printMatrix(matrix);

    // Definir y0 e condições de parada
    vector<double> y0 = defineInitialVector(matrix.size());
    double tolerance = 1e-5;
    int maxIterations = 1000;

    // Encontrar autovalor dominante
    pair<double, vector<double>> result = powerMethod(matrix, y0, tolerance, maxIterations);
    double dominantEigenvalue = result.first;
    vector<double> dominantEigenvector = result.second;

    cout << endl << "Autovalor dominante (lambda): " << dominantEigenvalue << endl;
    cout << "Autovetor dominante:" << endl;
    for (double val : dominantEigenvector) {
        cout << val << " ";
    }
    cout << endl;
    
    // Com o autovalor dominante, realizar 1000 deslocamentos no método das potências inverso
    cout << endl << "Encontrando todos os autovalores e autovetores usando deslocamento:" << endl;
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;

    double shiftStep = dominantEigenvalue / 1000;

    for (double shift = 0; shift <= dominantEigenvalue; shift += shiftStep) {
        pair<double, vector<double>> shiftedResult = inversePowerMethodWithShift(matrix, y0, shift, tolerance, maxIterations);
        double lambda = shiftedResult.first;
        vector<double> eigenvector = shiftedResult.second;

        eigenvalues.push_back(lambda);
        eigenvectors.push_back(eigenvector);
    }

    // Mapa para armazenar auto valores, contagem de ocorrências e a soma dos autovetores
    map<double, pair<int, vector<double>>> eigen_map;

    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        bool found = false;
        for (map<double, pair<int, vector<double>>>::iterator entry = eigen_map.begin(); entry != eigen_map.end(); ++entry) {
            if (is_equal(entry->first, eigenvalues[i])) {
                entry->second.first++; // Incrementa a contagem
                for (size_t j = 0; j < eigenvectors[i].size(); ++j) {
                    entry->second.second[j] += eigenvectors[i][j]; // Adiciona o autovetor à soma
                }
                found = true;
                break;
            }
        }
        if (!found) {
            eigen_map[eigenvalues[i]] = make_pair(1, eigenvectors[i]);
        }
    }

    // Calcula a média dos autovetores e armazena em um vetor
    vector<tuple<double, int, vector<double>>> eigen_info;
    for (map<double, pair<int, vector<double>>>::iterator entry = eigen_map.begin(); entry != eigen_map.end(); ++entry) {
        vector<double> avg_vector(entry->second.second.size());
        for (size_t i = 0; i < avg_vector.size(); ++i) {
            avg_vector[i] = entry->second.second[i] / entry->second.first; // Calcula a média
        }
        eigen_info.push_back(make_tuple(entry->first, entry->second.first, avg_vector));
    }

    // Ordena o vetor com base na contagem de ocorrências em ordem decrescente
    sort(eigen_info.begin(), eigen_info.end(), [](const tuple<double, int, vector<double>>& a, const tuple<double, int, vector<double>>& b) {
        return get<1>(a) > get<1>(b);
        });

    printEigenvaluesWithFrequency(eigen_map);

    // Seleciona os 7 principais auto valores com mais ocorrências
    vector<tuple<double, int, vector<double>>> top_eigen_info(eigen_info.begin(), eigen_info.begin() + min(eigen_info.size(), size_t(7)));

    // Ordena os 7 principais autovalores em ordem decrescente de autovalor
    sort(top_eigen_info.begin(), top_eigen_info.end(), [](const tuple<double, int, vector<double>>& a, const tuple<double, int, vector<double>>& b) {
        return get<0>(a) > get<0>(b);
        });

    // Determina o tamanho da matriz de acordo com o número de resultados em R (No nosso caso, 7)
    size_t matrix_size = top_eigen_info.size();
    vector<vector<double>> eigenvalue_matrix(matrix_size, vector<double>(matrix_size, 0.0));

    // Preenche a diagonal principal com os auto valores
    for (size_t i = 0; i < matrix_size; ++i) {
        eigenvalue_matrix[i][i] = get<0>(top_eigen_info[i]);
    }

    // Imprime a matriz de auto valores
    cout << "Matrix de auto valores:\n";
    printMatrix(eigenvalue_matrix);

    // Cria a matriz de auto vetores médios
    size_t vector_size = get<2>(top_eigen_info[0]).size();
    vector<vector<double>> vector_matrix(vector_size, vector<double>(matrix_size, 0.0));

    // Preenche a matriz de auto vetores médios
    for (size_t j = 0; j < matrix_size; ++j) {
        const vector<double>& avg_vector = get<2>(top_eigen_info[j]);
        for (size_t i = 0; i < vector_size; ++i) {
            vector_matrix[i][j] = avg_vector[i];
        }
    }

    // Imprime a matriz de autovetores médios
    cout << "\nMatriz de auto vetores medios em cada coluna:\n";
    printMatrix(vector_matrix);

    // Calcula o valor M para o critério de Kaiser
    int M = 0;
    for (size_t i = 0; i < matrix_size; ++i) {
        if (eigenvalue_matrix[i][i] > 1.0) {
            M++;
        }
    }

    // Imprime o valor M
    cout << "\nvalor M para o criterio de Kaiser: " << M << "\n";

    return 0;
}
