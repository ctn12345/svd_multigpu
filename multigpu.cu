#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include<omp.h>
#include <nvToolsExt.h>

#include "matrix_generate.hpp"
#include "large_matrix_svd.cu"
#include "small_matrix_svd.cu"

// #include "cusolver_svd.cu"

using namespace std;
using namespace std::chrono;
// fig 14 a
// 100x512x512 speedup over cusolver(CUDA platform)
void test17(){
    int gpu0=0,gpu1=1;
    int batch = 1;
    int height = 8192;
    int width = 8192;
    int th=0, tw=0;
    // int shape[3] = {batch, height, width};
    int minmn = height > width/2 ? width/2 : height;

    double* host_A = (double*)malloc(sizeof(double) * height * width);
    string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

    // read in host A
    FILE* A_fp = fopen(matrix_path1.data(), "r");
    if(A_fp==NULL){
        generate_matrix(height, width);
        A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            printf("open file falied\n");
            return ;
        }
    }
    for(int i=0; i < height*width; i++){
        fscanf(A_fp, "%lf", &host_A[i]);
    }

    fclose(A_fp);

    // steady_clock::time_point t1 = steady_clock::now();
    
    // double *dev_A;
    // cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    // for(int i=0; i<batch; i++){
    //     cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    // }
    // double *dev_U, *dev_V, *dev_diag;
    // cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    // cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    // cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);
    tw = 32;
    th = 32;
    int k = tw/2;
    int slice = th;
    int width_perdevice=width/2;

    // printf("input matrix shape: %d × %d × %d, tile shape: %d × %d\n", batch, height0, width0, th, tw);

// prams
// definition 1---gpu1
#pragma region
  
    /* p is the count of match-matrix A_ij, 
    e.g. a 16*16 matrix，k=4, 16*8 match-matrix A_ij's count is 2, i.e. p=2. */
    int p = (width_perdevice - 1) / (2 * k) + 1; 
    // each match-matrix A_ij is cut into slices at column wise, q is the count of these slices 
    int q = (height - 1) / slice + 1;
    // int width_perdevice = p * (2 * k);
    height = q * slice;
    int sliceNum = q;
    
    double* dev_A;  // fixed A
    double* dev_V;
    double* dev_V0;
    double* dev_U;
	int* dev_roundRobin; 
    
    double* dev_jointG;
    double* dev_Aij;

    double* dev_AiAi;   
    double* dev_AiAj;
    double* dev_AjAj;
    int* dev_pairsOfEVD;
    unsigned* host_allpass;
    unsigned* host_pass;
    unsigned* dev_allpass;
    unsigned* dev_pass;
    double *value;
    double* dev_swap_data;
    unsigned int *arr1;
    unsigned int *arr2;
    unsigned int *pairs;
    double *dev_norm;
    unsigned int *dev_order;
    double* host_Fnorm; 
    double* dev_tempFnorm;
    double* dev_Fnorm;
    double* dev_diag;
#pragma endregion
// definition 2 gpu
#pragma region
  
    /* p is the count of match-matrix A_ij, 
    e.g. a 16*16 matrix，k=4, 16*8 match-matrix A_ij's count is 2, i.e. p=2. */
    int p1 = (width_perdevice - 1) / (2 * k) + 1; 
    // each match-matrix A_ij is cut into slices at column wise, q is the count of these slices 
    int q1 = (height - 1) / slice + 1;
    // int width_perdevice = p * (2 * k);
    int height1 = q * slice;
    int sliceNum1 = q;
    
    double* dev_A_1;  // fixed A
    double* dev_V_1;
    double* dev_U_1;
    double* dev_V1;
	int* dev_roundRobin_1; 
    
    double* dev_jointG_1;
    double* dev_Aij_1;

    double* dev_AiAi_1;   
    double* dev_AiAj_1;
    double* dev_AjAj_1;
    int* dev_pairsOfEVD_1;
    unsigned* host_allpass_1;
    unsigned* host_pass_1;
    unsigned* dev_allpass_1;
    unsigned* dev_pass_1;
    double *value_1;
    double* dev_swap_data_1;
    unsigned int *arr1_1;
    unsigned int *arr2_1;
    unsigned int *pairs_1;
    double *dev_norm_1;
    unsigned int *dev_order_1;
    double* host_Fnorm_1; 
    double* dev_tempFnorm_1;
    double* dev_Fnorm_1;
    double* dev_diag_1;
#pragma endregion

// memory allocate


#pragma region
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaSetDevice(gpu1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    omp_set_num_threads(2);
#pragma endregion


#pragma region
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(gpuid == 0){
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width_perdevice * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMalloc((void **)&dev_V0, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMalloc((void **)&dev_diag,sizeof(double) * minmn);
            // dev_U = dev_U0;
            // dev_diag = dev_diag0;            
            // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

            cudaMalloc((void **)&dev_roundRobin, sizeof(int) * (2 * k - 1) * 2 * k);

            cudaMalloc((void **)&dev_jointG, sizeof(double) * 2*k * 2*k * p*batch);
            cudaMalloc((void **)&dev_Aij, sizeof(double) * height * 2*k * p*batch);

            cudaMalloc((void **)&dev_AiAi, sizeof(double) * k * k * sliceNum * p * batch);
            cudaMalloc((void **)&dev_AiAj, sizeof(double) * k * k * sliceNum * p * batch);
            cudaMalloc((void **)&dev_AjAj, sizeof(double) * k * k * sliceNum * p * batch);
            cudaMalloc((void **)&dev_pairsOfEVD, sizeof(int) * 2 * p * batch);
            // cudaMalloc((void **)&dev_swap_data,sizeof(double)*p*height*k);

            host_allpass = (unsigned *)malloc(sizeof(unsigned) * batch);
            host_pass = (unsigned *)malloc(sizeof(unsigned) * p * batch);
            cudaMalloc((void **)&dev_allpass, sizeof(unsigned) * batch);
            cudaMalloc((void **)&dev_pass, sizeof(unsigned) * p * batch);

            cudaMalloc((void **)&dev_norm, sizeof(double) * 2 * p * batch);
            cudaMalloc((void **)&dev_order, sizeof(unsigned int) * 2 * p * batch);
            host_Fnorm = (double *)malloc(sizeof(double) * batch);
            cudaMalloc((void **)&dev_tempFnorm, sizeof(double) * 2 * p * batch);
            cudaMalloc((void **)&dev_Fnorm, sizeof(double) * batch);

            cudaMemset(dev_V, 0, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
            // cudaMemset(dev_V0, 0, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMemset(dev_pairsOfEVD, 0, sizeof(int) * 2 * p * batch); 
            memset(host_pass, 0, sizeof(unsigned) * p * batch);
            cudaMemset(dev_pass, 0, sizeof(unsigned) * p * batch);
            cudaMemcpyAsync(dev_A,host_A,sizeof(double)*height*width_perdevice,cudaMemcpyHostToDevice,stream1);
        }
        else{
            cudaMalloc((void **)&dev_diag_1,sizeof(double) * minmn);
            cudaMalloc((void **)&dev_A_1, sizeof(double) * height * width_perdevice * batch);
            cudaMalloc((void **)&dev_V_1, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMalloc((void **)&dev_V1, sizeof(double) * width_perdevice * width_perdevice * batch);
            cudaMalloc((void **)&dev_U_1, sizeof(double) * height * height * batch);
            // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

            cudaMalloc((void **)&dev_roundRobin_1, sizeof(int) * (2 * k - 1) * 2 * k);

            cudaMalloc((void **)&dev_jointG_1, sizeof(double) * 2*k * 2*k * p*batch);
            cudaMalloc((void **)&dev_Aij_1, sizeof(double) * height * 2*k * p*batch);

            cudaMalloc((void **)&dev_AiAi_1, sizeof(double) * k * k * sliceNum * p1 * batch);
            cudaMalloc((void **)&dev_AiAj_1, sizeof(double) * k * k * sliceNum * p1 * batch);
            cudaMalloc((void **)&dev_AjAj_1, sizeof(double) * k * k * sliceNum * p1 * batch);
            cudaMalloc((void **)&dev_pairsOfEVD_1, sizeof(int) * 2 * p1 * batch);
            // cudaMalloc((void **)&dev_swap_data_1,sizeof(double)*p1*height*k);

            host_allpass_1 = (unsigned *)malloc(sizeof(unsigned) * batch);
            host_pass_1 = (unsigned *)malloc(sizeof(unsigned) * p1 * batch);
            cudaMalloc((void **)&dev_allpass_1, sizeof(unsigned) * batch);
            cudaMalloc((void **)&dev_pass_1, sizeof(unsigned) * p1 * batch);

            cudaMalloc((void **)&dev_norm_1, sizeof(double) * 2 * p1 * batch);
            cudaMalloc((void **)&dev_order_1, sizeof(unsigned int) * 2 * p1 * batch);
            host_Fnorm_1 = (double *)malloc(sizeof(double) * batch);
            cudaMalloc((void **)&dev_tempFnorm_1, sizeof(double) * 2 * p1 * batch);
            cudaMalloc((void **)&dev_Fnorm_1, sizeof(double) * batch);
            cudaMemcpyAsync(dev_A_1,host_A+height*width_perdevice,sizeof(double)*height*width_perdevice,cudaMemcpyHostToDevice,stream2);
        }
    }
#pragma endregion

// preset before svd  
#pragma region
    cudaSetDevice(gpu0);
    
    int shape[3]={batch,height,width_perdevice};
    double* host_A1,* host_A2;
    host_A1 = (double*)malloc(sizeof(double)*width_perdevice*height);
    host_A2 = (double*)malloc(sizeof(double)*width_perdevice*height);
    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 2.0;
    // omp thread

    double* swap_data_1 = (double*)malloc(sizeof(double)*p*height*k);
    double* swap_data_2 = (double*)malloc(sizeof(double)*p*height*k);
    clock_t start1,end;
    start1 = clock();
    double t1=0,t2=0,t3=0;
    clock_t begin1,end1;
    begin1 = clock();
    cudaSetDevice(gpu0);
    dim3 dimGrid0(1, 1, 1);
    dim3 dimBlock0(32, 32, 1);
    
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if (gpuid == 0)
            generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream1>>>(dev_roundRobin, 2*k);
        else
            generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream2>>>(dev_roundRobin_1, 2*k);
    }
    // generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream1>>>(dev_roundRobin, 2*k);
    // cudaSetDevice(gpu1);
    // generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream2>>>(dev_roundRobin_1, 2*k);
    // end1 = clock();
    // t3 += (double)(end1-begin1)/CLOCKS_PER_SEC;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
   
    
    double test_tag=0;
    int i = 0;
    while(test_tag < 1){    
        test_tag=4;    
        nvtxMarkA("Start SVD Computation"); // 这里会在 Nsight Systems 里显示一个标记
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            // printf("thread %d \n",gpuid);
            cudaSetDevice(gpuid);
            if(gpuid == 0){
                if(i != 0)
                    cudaStreamSynchronize(stream2);  // 等待 stream1 完
                // printf("thread %d \n",gpuid);
                svd_large_matrix_1(gpu0, stream1, false, dev_A, shape, dev_diag, dev_U, dev_V, dev_V0, 
                    th, tw, dev_roundRobin, dev_jointG, dev_Aij, dev_AiAi, dev_AiAj, 
                    dev_AjAj, dev_pairsOfEVD, dev_allpass, dev_pass, dev_norm, 
                    dev_order, dev_tempFnorm, dev_Fnorm,&test_tag);
                cudaMemcpyAsync(swap_data_1, dev_A + p * k * height, sizeof(double) * p * k * height, cudaMemcpyDeviceToHost, stream1);
            }
            else{
                if(i != 0)
                    cudaStreamSynchronize(stream1);
                // printf("thread %d ",gpuid);
                svd_large_matrix_1(gpu1, stream2, false, dev_A_1, shape, dev_diag_1, dev_U_1, dev_V_1, 
                    dev_V1, th, tw, dev_roundRobin_1, dev_jointG_1, dev_Aij_1, 
                    dev_AiAi_1, dev_AiAj_1, dev_AjAj_1, dev_pairsOfEVD_1, dev_allpass_1, 
                    dev_pass_1, dev_norm_1, dev_order_1, dev_tempFnorm_1, dev_Fnorm_1,&test_tag);
                cudaMemcpyAsync(swap_data_2, dev_A_1 + p1 * k * height, sizeof(double) * p1 * k * height, cudaMemcpyDeviceToHost, stream2);
                // cudaStreamSynchronize(stream1);  // 等待 stream1 完
                // cudaMemcpyAsync(dev_A_1 + p1 * k * height, swap_data_1, sizeof(double) * p * k * height, cudaMemcpyHostToDevice, stream2);
            }
            cudaSetDevice(gpuid);
            #pragma omp barrier  // **确保两个线程都完成了数据拷贝**
            if (gpuid == 0) {
                cudaMemcpyAsync(dev_A + p * k * height, swap_data_2, sizeof(double) * p1 * k * height, cudaMemcpyHostToDevice, stream1);
            }
            else {
                cudaMemcpyAsync(dev_A_1 + p1 * k * height, swap_data_1, sizeof(double) * p * k * height, cudaMemcpyHostToDevice, stream2);
            }
        }
        nvtxMarkA("End SVD Computation");

        // 第二轮 SVD 计算
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            if(gpuid == 0){ 
                cudaStreamSynchronize(stream2);  // 等待 stream2 完
                // printf("thread %d ",gpuid);
                svd_large_matrix_1(gpu0, stream1, false, dev_A, shape, dev_diag, dev_U, dev_V, dev_V0, 
                    th, tw, dev_roundRobin, dev_jointG, dev_Aij, dev_AiAi, dev_AiAj, 
                    dev_AjAj, dev_pairsOfEVD, dev_allpass, dev_pass, dev_norm, 
                    dev_order, dev_tempFnorm, dev_Fnorm,&test_tag);
                cudaMemcpyAsync(swap_data_1, dev_A + p * k * height, sizeof(double) * p * k * height, cudaMemcpyDeviceToHost, stream1);
                // cudaStreamSynchronize(stream2);
            }
            else{            
                cudaStreamSynchronize(stream1);  // 等待 stream1 完
                // printf("thread %d ",gpuid);
                svd_large_matrix_1(gpu1, stream2, false, dev_A_1, shape, dev_diag_1, dev_U_1, dev_V_1, 
                    dev_V1, th, tw, dev_roundRobin_1, dev_jointG_1, dev_Aij_1, 
                    dev_AiAi_1, dev_AiAj_1, dev_AjAj_1, dev_pairsOfEVD_1, dev_allpass_1, 
                    dev_pass_1, dev_norm_1, dev_order_1, dev_tempFnorm_1, dev_Fnorm_1,&test_tag);
                cudaMemcpyAsync(swap_data_2, dev_A_1, sizeof(double) * p1 * k * height, cudaMemcpyDeviceToHost, stream2);
                // cudaStreamSynchronize(stream1);
                
            }

            #pragma omp barrier  // **确保两个线程都完成了数据拷贝**

            if (gpuid == 0) {
                cudaMemcpyAsync(dev_A + p * k * height, swap_data_2, sizeof(double) * p1 * k * height, cudaMemcpyHostToDevice, stream1);
            }
            else {
                cudaMemcpyAsync(dev_A_1, swap_data_1, sizeof(double) * p * k * height, cudaMemcpyHostToDevice, stream2);
            }
        }


        bool flag = false;

        // printf("third\n");
        // 第三轮 SVD 计算
        cudaEventRecord(start, stream1);
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            if(gpuid == 0){
                cudaStreamSynchronize(stream2);
                // printf("thread %d ",gpuid);
                svd_large_matrix_1(gpu0, stream1, flag, dev_A, shape, dev_diag, dev_U, dev_V, dev_V0, 
                    th, tw, dev_roundRobin, dev_jointG, dev_Aij, dev_AiAi, dev_AiAj, 
                    dev_AjAj, dev_pairsOfEVD, dev_allpass, dev_pass, dev_norm, 
                    dev_order, dev_tempFnorm, dev_Fnorm,&test_tag);
                // 只有 i != 3 时进行数据交换
                if(i != 3){
                    cudaMemcpyAsync(swap_data_1, dev_A + p * k * height, sizeof(double) * p * k * height, cudaMemcpyDeviceToHost, stream1);
                }  
            }
            else{
                cudaStreamSynchronize(stream1);
                svd_large_matrix_1(gpu1, stream2, flag, dev_A_1, shape, dev_diag_1, dev_U_1, dev_V_1, 
                    dev_V1, th, tw, dev_roundRobin_1, dev_jointG_1, dev_Aij_1, 
                    dev_AiAi_1, dev_AiAj_1, dev_AjAj_1, dev_pairsOfEVD_1, dev_allpass_1, 
                    dev_pass_1, dev_norm_1, dev_order_1, dev_tempFnorm_1, dev_Fnorm_1,&test_tag);
                // 只有 i != 3 时进行数据交换
                if(i != 3){
                    cudaMemcpyAsync(swap_data_2, dev_A_1 + p1 * k * height, sizeof(double) * p1 * k * height, cudaMemcpyDeviceToHost, stream2);     
                }  
            }
           if(i != 3){
                #pragma omp barrier
                if (gpuid == 0) {
                    cudaMemcpyAsync(dev_A + p * k * height, swap_data_2, sizeof(double) * p1 * k * height, cudaMemcpyHostToDevice, stream1);
                }
                else {
                    cudaMemcpyAsync(dev_A_1 + p1 * k * height, swap_data_1, sizeof(double) * p * k * height, cudaMemcpyHostToDevice, stream2);
                }
            }  
        }
        ++i; 
    }
    #pragma omp parallel
    {
        dim3 dimGrid10(2 * p, batch, 1);
        dim3 dimBlock10(32, k, 1);
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(gpuid == 0){
            getUDV<<<dimGrid10, dimBlock10,0,stream1>>>(dev_A, dev_U, dev_V, dev_V0, height, width, height, width, p, height/32, dev_diag, width, k);  //&1.3
        }
        else{
            getUDV<<<dimGrid10, dimBlock10,0,stream2>>>(dev_A_1, dev_U_1, dev_V_1, dev_V1, height, width, height, width, p, height/32, dev_diag_1, width, k);  //&1.3
        }
    }
    end1 = clock();
    printf("it costs %lfs",(double)(end1-begin1)/CLOCKS_PER_SEC);
    
    double* host_diag = (double*)malloc(sizeof(double)*minmn*batch);
    double* host_diag1 = (double*)malloc(sizeof(double)*minmn*batch);
    cudaSetDevice(gpu0);
    cudaMemcpy(host_diag,dev_diag,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    cudaSetDevice(gpu1);
    cudaMemcpy(host_diag1,dev_diag_1,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    FILE* file2 = fopen("dev_diag.txt","w");
    for(int f = 0;f < minmn;++f){
        fprintf(file2,"%lf %lf ",host_diag[f],host_diag1[f]);
    }
    
    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

int main(int argc, char* argv[]){
    test17();
}