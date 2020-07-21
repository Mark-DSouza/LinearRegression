#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

typedef struct MATRIX {
    double *matrix;
    int rows;
    int cols;
} Matrix;

void init(Matrix *mat, int rows, int columns) {
    mat -> matrix = (double *)malloc(rows * columns * sizeof(double));
    mat -> rows = rows;
    mat -> cols = columns;
}

void del(Matrix *mat) {
    free(mat -> matrix);
    free(mat);
}

Matrix *scaMul(double scalar, Matrix *mat) {
    int rows = mat -> rows;
    int cols = mat -> cols;

    Matrix *res = (Matrix *)malloc(sizeof(Matrix));
    init(res, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            res -> matrix[i * cols + j] 
                = scalar * mat -> matrix[i * cols + j];
    return res;
}

Matrix *matAdd(Matrix *mat1, Matrix *mat2, int sign) {
    if (mat1 -> rows != mat2 -> rows || mat1 -> cols != mat2 -> cols) {
        printf("\nDimensions not compatible for matrix addition!\n");
        return NULL;
    }

    int rows = mat1 -> rows;
    int cols = mat1 -> cols;

    Matrix *res = (Matrix *)malloc(sizeof(Matrix));
    init(res, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            res -> matrix[i * cols + j] 
                = mat1 -> matrix[i * cols + j] + mat2 -> matrix[i * cols + j] * sign;
    return res;
}

Matrix *matMul(Matrix *mat1, Matrix *mat2) {
    if (mat1 -> rows != mat2 -> cols) {
        printf("\nDimensions not compatible for matrix multiplication!\n");
        return NULL;
    }
    
    int rows = mat1 -> rows;
    int common = mat1 -> cols;
    int cols = mat2 -> cols;
    
    Matrix *res = (Matrix *)malloc(sizeof(Matrix));
    init(res, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            res -> matrix[i * cols + j] = 0;
            for (int k = 0; k < common; k++)
                res -> matrix[i * cols + j] 
                    += mat1 -> matrix[i * cols + k] * mat2 -> matrix[k * cols + j];
        }
    
    return res; 
}

Matrix *matTrans(Matrix *mat) {
    int rows = mat -> cols;
    int cols = mat -> rows;

    Matrix *trans = (Matrix *)malloc(sizeof(Matrix));
    init(trans, rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            trans -> matrix[i * cols + j] = mat -> matrix[j * cols + i];
    
    return trans;
}

void matDisplay(Matrix *mat) {
    int rows = mat -> rows;
    int cols = mat -> cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%lf\t", mat -> matrix[i * cols + j]);
        printf("\n");
    }
}

double computeCost(Matrix *HypoMinusY, int m) {
    double cost = 0;
    for (int i = 0; i < m; i++)
        cost += pow(HypoMinusY -> matrix[i], 2);
    cost *= 0.5 * m;
    
    return cost;
}

Matrix *gradientDescent(Matrix *X, Matrix *y, Matrix *theta, 
                            int alpha, int iterations, int m) 
{
    Matrix *Hypo;
    Matrix *HypoMinusY;
    Matrix *XTrans;
    Matrix *Dow;
    Matrix *AlphaDow;
    Matrix *temp;
    
    for (int i = 1; i <= iterations; i++) {
        Hypo = matMul(X, theta);
        HypoMinusY = matAdd(Hypo, y, -1);
        XTrans = matTrans(X);
        Dow = matMul(XTrans, HypoMinusY);
        AlphaDow = scaMul(alpha / m, Dow);

        temp = theta;
        theta = matAdd(theta, AlphaDow, -1);
        if (!(i % 5))
            printf("Iteration = %d; Cost = %lf\n", i, computeCost(HypoMinusY, m));
        
        del(Hypo);
        del(HypoMinusY);
        del(XTrans);
        del(Dow);
        del(AlphaDow);
        del(temp);
    }

    return theta;
}

void getDataset(Matrix *X, Matrix *y, Matrix *theta,int m, int n, char *fileName) {
    FILE *fp = fopen(fileName, "r");
     
    for (int i = 0; i < m; i++){
        X -> matrix[i * (n + 1)] = 1;
        
        for (int j = 1; j < n + 1; j++) {
            fscanf(fp, "%lf%*c", &(X -> matrix[i * (n + 1) + j]));
        }
        fscanf(fp, "%lf", &(y -> matrix[i]));
    }
    
    fclose(fp);
}

void getUserQuery(Matrix *query, int n) {
    printf("\nEnter the features of the required prediction:");
    query->matrix[0] = 1;
    for (int i = 1; i <= n; i++)
        scanf("%lf", &(query->matrix[i]));
}

void displayPrediction(Matrix *theta, Matrix *query) {
    Matrix *res = (Matrix *)malloc(sizeof(Matrix));
    res = matMul(query, theta);

    printf("\nThe prediction for the features  ");
    for (int i = 1; i <= query -> cols; i++)
        printf("%lf  ", query -> matrix[i]);
    printf("is -> %lf\n", res -> matrix[0]);
}

void linearRegression(double alpha, int iterations, int m, int n, char *fileName) {
    Matrix *X = (Matrix *)malloc(sizeof(Matrix));
    Matrix *y = (Matrix *)malloc(sizeof(Matrix));
    Matrix *theta = (Matrix *)malloc(sizeof(Matrix));

    init(X, m, n+1);
    init(y, m, 1);
    init(theta, n+1, 1);
    
    getDataset(X, y, theta, m, n, fileName);

    for (int i = 0; i < n + 1; i++)
        theta -> matrix[i] = 0;
    
    theta = gradientDescent(X, y, theta, alpha, iterations, m);
    printf("Theta obtained after %d iterations of Gradient Descent:\n", iterations);
    matDisplay(theta);

    Matrix *query = (Matrix *)malloc(sizeof(Matrix));
    init(query, 1, n+1);

    char choice;
    do {
        getUserQuery(query, n);
        displayPrediction(theta, query);
        printf("\nWould you like another prediction?[y/n]: ");
        scanf(" %c", &choice);
    } while(toupper(choice) == 'Y');

    del(X);
    del(y);
    del(theta);
    del(query);
}

void getUserInput(double *alpha, int *iterations, int *m, int *n, char *fileName) {
    printf("\nEnter the value of the learning rate: ");
    scanf("%lf", alpha);
    printf("\nEnter the number of iterations: ");
    scanf("%d", iterations);
    printf("\n---Prepare to enter information about dataset---");
    printf("\nEnter the number of training examples: ");
    scanf("%d", m);
    printf("\nEnter the number of features: ");
    scanf("%d", n);
    printf("\nEnter the file name of the dataset: ");
    scanf(" %s", fileName);
}

int main() {
    printf("\n=====Welcome to the Linear Regression Program=====\n\n\n");
    
    double alpha;
    int iterations, m, n;
    char choice, fileName[100];
    
    do {    
        getUserInput(&alpha, &iterations, &m, &n, fileName);
        
        linearRegression(alpha, iterations, m, n, fileName);

        printf("Do you want to perform another run of Linear Regression?[y/n]: ");
        scanf(" %c", &choice);
    } while(toupper(choice) == 'Y');
    
    return 0;
}