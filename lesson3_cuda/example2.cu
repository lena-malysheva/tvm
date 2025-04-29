#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define M 1024
#define N 4096
#define K 4096

__global__ void matrixMulVectorized(float4 *A, float4 *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n/4; i++) {
            float4 a = A[row * n/4 + i];
            float4 b = B[i * k/4 + col/4];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        C[row * k + col] = sum;
    }
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    C = (float*)malloc(size_C);

    for (int i = 0; i < M * N; i++) A[i] = 3.0f;
    for (int i = 0; i < N * K; i++) B[i] = 2.0f;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMulVectorized<<<blocksPerGrid, threadsPerBlock>>>(
        (float4*)d_A, (float4*)d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
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

    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}