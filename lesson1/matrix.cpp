#include <iostream>
#include <cstdio>
#include <chrono>
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

void multiplyv1(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            c[m * N + n] = 0;
            for (int k = 0; k < K; k++)
                c[m * N + n] += a[m * K + k] * b[k * N + n];
        }
    }
}

void multiplyv2(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n1 = 0; n1 < N; n1 += 8) {//разделить оси н делает сплит
        for (int n2 = 0; n2 < 8; n2++){
            c[m * N + (n1 + n2)] = 0;
            for (int k = 0; k < K; k++)
                c[m * N + (n1 + n2)] += a[m * K + k] * b[k * N + (n1 + n2)];
        }
        }
    }
}

void multiplyv3(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n1 = 0; n1 < N; n1 += 8) {
            for (int n2 = 0; n2 < 8; n2++)
                c[m * N + (n1 + n2)] = 0;
            for (int k = 0; k < K; k++)
                for (int n2 = 0; n2 < 8; n2++){
                    c[m * N + (n1 + n2)] += a[m * K + k] * b[k * N + (n1 + n2)];}
                }
    }
}

void multiplyv4(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    for (int m1 = 0; m1 < M; m1 += 8) {
        
            for (int n1 = 0; n1 < N; n1 += 8) {
                for (int n2 = 0; n2 < 8; n2++)
                    for (int m2 = 0; m2 < 8; m2++)
                        c[(m1 + m2) * N + (n1 + n2)] = 0;
                for (int k = 0; k < K; k++)
                    for (int n2 = 0; n2 < 8; n2++)
                        for (int m2 = 0; m2 < 8; m2++)
                            c[(m1 + m2) * N + (n1 + n2)] += a[(m1 + m2) * K + k] * b[k * N + (n1 + n2)];
                }
    }
}

void multiplyv5(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n1 = 0; n1 < N; n1 += 16) {
            for (int n2 = 0; n2 < 16; n2++)
                c[m * N + n1 + n2] = 0;
            for (int k = 0; k < K; k++)
                for (int n2 = 0; n2 < 16; n2++)
                    c[m * N + n1 + n2] += a[m * K + k] * b[k * N + n1 + n2];
        }
    }
}


int main()
{
    int M = 4096;
    int K = 128;
    int N = 1024;
    
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* c = new float[M * N];

    for (int m = 0; m < M; m++) 
        for (int k = 0; k < K; k++) 
        { 
            a[m * K + k] = rand() % 10;
        }

    for (int k = 0; k < K; k++) 
        for (int n = 0; n < N; n++) 
        { 
            b[k * N + n] = rand() % 10;
        }
        
    #if defined(AVX) 
        std::cout << "AVX is supported.\n"; 
    #else 
        std::cout << "AVX is not supported.\n"; 
    #endif
        Time::time_point time1_ = Time::now();

    multiplyv1(a, b, c, M, K, N);

    Time::time_point time2_ = Time::now();
    std::cout << "Matix multiplication: " << std::chrono::duration_cast<ns>(time2_ - time1_).count() * 0.000001 << " ms" << std::endl;

    Time::time_point time1 = Time::now();

    multiplyv5(a, b, c, M, K, N);

    Time::time_point time2 = Time::now();
    std::cout << "Matix multiplication: " << std::chrono::duration_cast<ns>(time2 - time1).count() * 0.000001 << " ms" << std::endl;

    /*for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << c[i * n +j] << "\t";
        }
        std::cout << "\n";
    }*/
    delete [] a;  

    delete [] b; 
    
    delete [] c; 

    return 0;
}