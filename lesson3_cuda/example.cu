#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// размеры матриц: A[M][N] * B[N][K] = C[M][K]
#define M 1024
#define N 4096
#define K 4096

// ядро для умножения матриц
__global__ void matrixMul(float *A, float *B, float *C) {//показывает, что это kernel, который выполняется на gpu, но вызывается с cpu
    int row = blockIdx.y * blockDim.y + threadIdx.y;//blockIdx.x, blockIdx.y — координаты блока в сетке. threadIdx.x, threadIdx.y — координаты потока в блоке
    int col = blockIdx.x * blockDim.x + threadIdx.x;//blockDim.x, blockDim.y — размеры блока

    if (row < M && col < K) {//проверка границ матрицы
        float sum = 0.0f;//накапливает результат
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];//каждый поток выполняет скалярное произведение строки матрицы A и столбца матрицы B
        }
        C[row * K + col] = sum;//значение sum записывает в C с координатами (row, col)
    }
}

int main() {
    float *A, *B, *C;       // хост-матрицы
    float *d_A, *d_B, *d_C; // девайс-матрицы
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    // выделение память на хосте
    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    C = (float*)malloc(size_C);

    for (int i = 0; i < M * N; i++) A[i] = 3.0f; 
    for (int i = 0; i < N * K; i++) B[i] = 2.0f; 

    // выделение памяти на устройстве
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // копирование данных на устройство
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // задаются размеры блоков и грида
    dim3 threadsPerBlock(64, 8);//создает структуру dim3, определяющую размер блока потоков в виде 2D-сетки,  по X,  по Y
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,//по X(столбцы)
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y//по Y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // запуск ядра
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);//blocksPerGrid количество блоков в сетке, threadsPerBlock количество потоков в каждом блоке

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // ждет завершения всех операций

    // вычисление времени
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // копирование результата обратно
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);//C указатель на матрицу в cpu, d_C на матрицу в GPU, cudaMemcpyDeviceToHost гапрвление копирования
    printf("Время выполнения: %.3f мс\n", milliseconds);
    printf("C[0][0] = %f \n", C[0]);

    bool allElementsCorrect = true;
    float expectedValue = N * 6.0f;
    float tolerance = 1e-5f;

    for (int i = 0; i < M * K; i++) {
        if (fabs(C[i] - expectedValue) > tolerance) {
            printf("Ошибка: C[%d] = %f (ожидалось %f)\n", i, C[i], expectedValue);
            allElementsCorrect = false;
            break;
        }
    }

    if (allElementsCorrect) {
        printf("Все элементы матрицы C корректны (равны %f)\n", expectedValue);
    }

    // освобождение памяти
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
