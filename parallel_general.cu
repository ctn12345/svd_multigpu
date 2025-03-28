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

void fill_hostorder_total(int* host_order_total,int** host_order,double** host_norm,int* p,int num_gpus){
    int* idx = (int*)malloc(sizeof(int)*num_gpus);
    memset(idx, 0, sizeof(int) * num_gpus);
    bool flag = false;
    int index = 0;
    while(!flag){
        double tmp = host_norm[0][idx[0]];
        int need_index = 0;
        for(int u = 0;u < num_gpus;++u){
            if(idx[u] < 2 * p[u] && tmp > host_norm[u][idx[u]]){
                tmp = host_norm[u][idx[u]];
                need_index = u;
            }
        }
        idx[need_index]++;
        host_order_total[index] = host_order[need_index][idx[need_index]]+p[0]*2*need_index;
        index++;
        flag = true;
        for(int u = 0;u < num_gpus;++u){
            if(idx[u] < 2*p){
                flag = false;
            }
        }
    }
    // int index=0,idx1=0,idx2=0;
    // while(idx1 < 2*p || idx2 < 2*p1){
    //     if(idx1 < 2*p && idx2 < 2*p1){
    //         if(host_norm_1[host_order_1[idx1]] < host_norm_2[host_order_2[idx2]]){
    //             host_order_total[index]=host_order_1[idx1];
    //             idx1++;
    //             index++;
    //         }
    //         else{
    //             host_order_total[index] = host_order_2[idx2]+2*p;
    //             idx2++;
    //             index++;
    //         }
    //     }
    //     else if(idx1 < 2*p){
    //         host_order_total[index] = host_order_1[idx1];
    //         idx1++;
    //         index++;
    //     }
    //     else{
    //         host_order_total[index] = host_order_2[idx2]+2*p;
    //         idx2++;
    //         index++;
    //     }
    // }
}
// fig 14 a
// 100x512x512 speedup over cusolver(CUDA platform)
void test17(){
    int num_gpus;

    // 获取 GPU 数量
    cudaGetDeviceCount(&num_gpus);
    int gpu0=0,gpu1=1;
    int batch = 1;
    int height = 1024;
    int width = 1024;
    int th=0, tw=0;
    // int shape[3] = {batch, height, width};
    int minmn = height > width/num_gpus ? width/num_gpus : height;

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

    tw = 32;
    th = 32;
    int k = tw/2;
    int slice = th;
    int width_perdevice=width/num_gpus;
    size_t pitch;
    int*p_a,*p_b,*p_ab;

    // printf("input matrix shape: %d × %d × %d, tile shape: %d × %d\n", batch, height0, width0, th, tw);

// prams
// definition 1---gpu1
#pragma region
  
    /* p is the count of match-matrix A_ij, 
    e.g. a 16*16 matrix，k=4, 16*8 match-matrix A_ij's count is 2, i.e. p=2. */
    // int p[num_gpus] = {(width_perdevice - 1) / (2 * k) + 1,(width_perdevice - 1) / (2 * k) + 1,(width_perdevice - 1) / (2 * k) + 1,(width_perdevice - 1) / (2 * k) + 1}; 
    // each match-matrix A_ij is cut into slices at column wise, q is the count of these slices 
    // int q[num_gpus] = {(height - 1) / slice + 1,(height - 1) / slice + 1,(height - 1) / slice + 1,(height - 1) / slice + 1};
    // int width_perdevice = p * (2 * k);
    // height = q * slice;
    // 申请指针数组（动态分配）
    int** dev_pa = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pb = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pab = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pa1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pb1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pab1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pa_1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pb_1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pab_1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pa1_1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pb1_1 = (int**)malloc(num_gpus * sizeof(int*));
    int** dev_pab1_1 = (int**)malloc(num_gpus * sizeof(int*));

    int* p = (int*)malloc(num_gpus * sizeof(int));
    int* q = (int*)malloc(num_gpus * sizeof(int));
    int* sliceNum = (int*)malloc(num_gpus * sizeof(int));

    double** dev_A = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_V = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_V0 = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_U = (double**)malloc(num_gpus * sizeof(double*));
    int** dev_roundRobin = (int**)malloc(num_gpus * sizeof(int*));

    double** dev_jointG = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_Aij = (double**)malloc(num_gpus * sizeof(double*));

    double** dev_AiAi = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_AiAj = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_AjAj = (double**)malloc(num_gpus * sizeof(double*));

    int** dev_pairsOfEVD = (int**)malloc(num_gpus * sizeof(int*));
    unsigned** host_allpass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    unsigned** host_pass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    unsigned** dev_allpass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    unsigned** dev_pass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));

    double** value = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_swap_data = (double**)malloc(num_gpus * sizeof(double*));

    unsigned int** arr1 = (unsigned int**)malloc(num_gpus * sizeof(unsigned int*));
    unsigned int** arr2 = (unsigned int**)malloc(num_gpus * sizeof(unsigned int*));
    unsigned int** pairs = (unsigned int**)malloc(num_gpus * sizeof(unsigned int*));

    double** dev_norm = (double**)malloc(num_gpus * sizeof(double*));
    unsigned int** dev_order = (unsigned int**)malloc(num_gpus * sizeof(unsigned int*));

    double** host_Fnorm = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_tempFnorm = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_Fnorm = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_diag = (double**)malloc(num_gpus * sizeof(double*));

    double** host_A_per = (double**)malloc(num_gpus * sizeof(double*));
    int** host_order = (int**)malloc(num_gpus * sizeof(int*));
    // int** host_index = (int*)malloc(num_gpus*sizeof(int*));
    double** host_rawnorm = (double**)malloc(num_gpus*sizeof(double*));
    double** host_norm = (double**)malloc(num_gpus*sizeof(double*));
    double** host_swap_data = (double**)malloc(sizeof(double*)*num_gpus);
    double** test_Fnorm = (double**)malloc(sizeof(double*)*num_gpus);
#pragma endregion

#pragma region
omp_set_num_threads(num_gpus);
#pragma omp parallel
{
    int gpuid = omp_get_thread_num();
    p[gpuid]=(width_perdevice-1)/(2*k)+1;
    cudaMalloc((void**)&dev_pa[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pb[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab[gpuid],sizeof(int)*2*p[gpuid]*(2*p[gpuid]-1));
    // next_time
    cudaMalloc((void**)&dev_pa1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pb1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab1[gpuid],sizeof(int)*2*p[gpuid]*p[gpuid]);

    cudaMalloc((void **)&dev_U[gpuid], sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_A[gpuid], sizeof(double) * height * width_perdevice * batch);
    cudaMalloc((void **)&dev_V[gpuid], sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMalloc((void **)&dev_V0[gpuid], sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMalloc((void **)&dev_diag[gpuid],sizeof(double) * minmn);
    // dev_U = dev_U0;
    // dev_diag = dev_diag0;

    
    // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

    cudaMalloc((void **)&dev_roundRobin[gpuid], sizeof(int) * (2 * k - 1) * 2 * k);

    cudaMalloc((void **)&dev_jointG[gpuid], sizeof(double) * 2*k * 2*k * p[gpuid]*batch);
    cudaMalloc((void **)&dev_Aij[gpuid], sizeof(double) * height * 2*k * p[gpuid]*batch);

    cudaMalloc((void **)&dev_AiAi[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_AiAj[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_AjAj[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_pairsOfEVD[gpuid], sizeof(int) * 2 * p * batch);
    cudaMalloc((void **)&dev_swap_data[gpuid],sizeof(double)*p[gpuid]*height*k);
    cudaMemset(dev_V[gpuid], 0, sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMemset(dev_U[gpuid], 0, sizeof(double) * height * height * batch);
    // cudaMemset(dev_V0, 0, sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMemset(dev_pairsOfEVD[gpuid], 0, sizeof(int) * 2 * p[gpuid] * batch); 
    cudaMemset(dev_pass[gpuid], 0, sizeof(unsigned) * p[gpuid] * batch);
    memset(host_pass[gpuid], 0, sizeof(unsigned) * p[gpuid] * batch);
    host_A_per[gpuid] = (double*)malloc(sizeof(double)*width_perdevice*height);
    host_order[gpuid] = (int*)malloc(sizeof(int)*num_gpus*p[gpuid]*2*batch);
    host_rawnorm[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*2*batch);
    host_norm[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*2*batch);
    host_swap_data[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*height*k);
    test_Fnorm[gpuid] = (double*)malloc(sizeof(double)*batch);
}
#pragma endregion
#pragma region
// definition 2 gpu
// #pragma region
  
//     /* p is the count of match-matrix A_ij, 
//     e.g. a 16*16 matrix，k=4, 16*8 match-matrix A_ij's count is 2, i.e. p=2. */
//     int p1 = (width_perdevice - 1) / (2 * k) + 1; 
//     // each match-matrix A_ij is cut into slices at column wise, q is the count of these slices 
//     int q1 = (height - 1) / slice + 1;
//     // int width_perdevice = p * (2 * k);
//     int height1 = q * slice;
//     int sliceNum1 = q;
    
//     double* dev_A_1;  // fixed A
//     double* dev_V_1;
//     double* dev_V1;
//     double* dev_U_1;
// 	int* dev_roundRobin_1; 
    
//     double* dev_jointG_1;
//     double* dev_Aij_1;

//     double* dev_AiAi_1;   
//     double* dev_AiAj_1;
//     double* dev_AjAj_1;
//     int* dev_pairsOfEVD_1;
//     unsigned* host_allpass_1;
//     unsigned* host_pass_1;
//     unsigned* dev_allpass_1;
//     unsigned* dev_pass_1;
//     double *value_1;
//     double* dev_swap_data_1;
//     unsigned int *arr1_1;
//     unsigned int *arr2_1;
//     unsigned int *pairs_1;
//     double *dev_norm_1;
//     unsigned int *dev_order_1;
//     double* host_Fnorm_1; 
//     double* dev_tempFnorm_1;
//     double* dev_Fnorm_1;
//     double* dev_diag_1;
// #pragma endregion

// memory allocate


// #pragma region
//     cudaSetDevice(gpu0);
//     cudaMalloc((void**)&p_a,sizeof(int)*(p+p1));
//     cudaMalloc((void**)&p_b,sizeof(int)*(p+p1));
//     cudaMalloc((void**)&p_ab,sizeof(int)*2*(p+p1)*(2*(p+p1)-1));
//     cudaMalloc((void**)&dev_pa,sizeof(int)*(p));
//     cudaMalloc((void**)&dev_pb,sizeof(int)*(p));
//     cudaMalloc((void**)&dev_pab,sizeof(int)*2*p*(2*p-1));
//     // next_time
//     cudaMalloc((void**)&dev_pa1,sizeof(int)*p);
//     cudaMalloc((void**)&dev_pb1,sizeof(int)*p);
//     cudaMalloc((void**)&dev_pab1,sizeof(int)*2*p*p);

//     cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
//     cudaMalloc((void **)&dev_A, sizeof(double) * height * width_perdevice * batch);
//     cudaMalloc((void **)&dev_V, sizeof(double) * width_perdevice * width_perdevice * batch);
//     cudaMalloc((void **)&dev_V0, sizeof(double) * width_perdevice * width_perdevice * batch);
//     cudaMalloc((void **)&dev_diag,sizeof(double) * minmn);
//     // dev_U = dev_U0;
//     // dev_diag = dev_diag0;

    
//     // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

//     cudaMalloc((void **)&dev_roundRobin, sizeof(int) * (2 * k - 1) * 2 * k);

//     cudaMalloc((void **)&dev_jointG, sizeof(double) * 2*k * 2*k * p*batch);
//     cudaMalloc((void **)&dev_Aij, sizeof(double) * height * 2*k * p*batch);

//     cudaMalloc((void **)&dev_AiAi, sizeof(double) * k * k * sliceNum * p * batch);
//     cudaMalloc((void **)&dev_AiAj, sizeof(double) * k * k * sliceNum * p * batch);
//     cudaMalloc((void **)&dev_AjAj, sizeof(double) * k * k * sliceNum * p * batch);
//     cudaMalloc((void **)&dev_pairsOfEVD, sizeof(int) * 2 * p * batch);
//     cudaMalloc((void **)&dev_swap_data,sizeof(double)*p*height*k);

//     host_allpass = (unsigned *)malloc(sizeof(unsigned) * batch);
//     host_pass = (unsigned *)malloc(sizeof(unsigned) * p * batch);
//     cudaMalloc((void **)&dev_allpass, sizeof(unsigned) * batch);
//     cudaMalloc((void **)&dev_pass, sizeof(unsigned) * p * batch);

//     cudaMalloc((void **)&dev_norm, sizeof(double) * 2 * p * batch);
//     cudaMalloc((void **)&dev_order, sizeof(unsigned int) * 2 * p * batch);
//     host_Fnorm = (double *)malloc(sizeof(double) * batch);
//     cudaMalloc((void **)&dev_tempFnorm, sizeof(double) * 2 * p * batch);
//     cudaMalloc((void **)&dev_Fnorm, sizeof(double) * batch);

// #pragma endregion
// #pragma region
//     cudaSetDevice(gpu1);
//     cudaMalloc((void **)&dev_diag_1,sizeof(double) * minmn);
//     cudaMalloc((void **)&dev_A_1, sizeof(double) * height * width_perdevice * batch);
//     cudaMalloc((void **)&dev_V_1, sizeof(double) * width_perdevice * width_perdevice * batch);
//     cudaMalloc((void **)&dev_V1, sizeof(double) * width_perdevice * width_perdevice * batch);
//     cudaMalloc((void **)&dev_U_1, sizeof(double) * height * height * batch);

//     cudaMalloc((void**)&dev_pa_1,sizeof(int)*(p1));
//     cudaMalloc((void**)&dev_pb_1,sizeof(int)*(p1));
//     cudaMalloc((void**)&dev_pab_1,sizeof(int)*2*p1*(2*p1-1));
//     cudaMalloc((void**)&dev_pa1_1,sizeof(int)*(p1));
//     cudaMalloc((void**)&dev_pb1_1,sizeof(int)*(p1));
//     cudaMalloc((void**)&dev_pab1_1,sizeof(int)*2*p1*p1);
//     // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

//     cudaMalloc((void **)&dev_roundRobin_1, sizeof(int) * (2 * k - 1) * 2 * k);

//     cudaMalloc((void **)&dev_jointG_1, sizeof(double) * 2*k * 2*k * p*batch);
//     cudaMalloc((void **)&dev_Aij_1, sizeof(double) * height * 2*k * p*batch);

//     cudaMalloc((void **)&dev_AiAi_1, sizeof(double) * k * k * sliceNum * p1 * batch);
//     cudaMalloc((void **)&dev_AiAj_1, sizeof(double) * k * k * sliceNum * p1 * batch);
//     cudaMalloc((void **)&dev_AjAj_1, sizeof(double) * k * k * sliceNum * p1 * batch);
//     cudaMalloc((void **)&dev_pairsOfEVD_1, sizeof(int) * 2 * p1 * batch);
//     cudaMalloc((void **)&dev_swap_data_1,sizeof(double)*p1*height*k);

//     host_allpass_1 = (unsigned *)malloc(sizeof(unsigned) * batch);
//     host_pass_1 = (unsigned *)malloc(sizeof(unsigned) * p1 * batch);
//     cudaMalloc((void **)&dev_allpass_1, sizeof(unsigned) * batch);
//     cudaMalloc((void **)&dev_pass_1, sizeof(unsigned) * p1 * batch);

//     cudaMalloc((void **)&dev_norm_1, sizeof(double) * 2 * p1 * batch);
//     cudaMalloc((void **)&dev_order_1, sizeof(unsigned int) * 2 * p1 * batch);
//     host_Fnorm_1 = (double *)malloc(sizeof(double) * batch);
//     cudaMalloc((void **)&dev_tempFnorm_1, sizeof(double) * 2 * p1 * batch);
//     cudaMalloc((void **)&dev_Fnorm_1, sizeof(double) * batch);

// #pragma endregion

// preset before svd  
#pragma endregion
#pragma region
    cudaSetDevice(gpu0);
    
   
    
    int shape[3]={batch,height,width_perdevice};
    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 2.0;

    cudaStream_t* stream = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(gpuid);
        cudaMemcpyAsync(dev_A[gpuid],host_A+i*width_perdevice*height,cudaMemcpyHostToDevice,stream[gpuid]);
    }
    dim3 dimGrid0(1, 1, 1);
    dim3 dimBlock0(32, 32, 1);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream[gpuid]>>>(dev_roundRobin[gpuid], 2*k);
    }
    
    
    bool continue_flag = false;

    omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        getRankNewNew_2<<<1,1024,0,stream[gpuid]>>>(2*p[gpuid],dev_pab[gpuid],dev_pa[gpuid],dev_pb[gpuid]);
        getRankNewNew_1<<<1,1024,0,stream[gpuid]>>>(p[gpuid],dev_pa1[gpuid],dev_pb1[gpuid],dev_pab1[gpuid]);
    }
    int sweep = 0,maxsweep = 20;
    double svd_tol = 1e-7;
    int* raw_host_order = (int*)malloc(sizeof(int)*2*p[0]*num_gpus*batch);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        // printf("gpuid %d \n",gpuid);
        compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width_perdevice, p[gpuid], q, k);
        binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);
        
    } 
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(host_order[i],dev_order[i],sizeof(int)*2*p[gpuid]*batch,cudaMemcpyDeviceToHost,stream[i]);
        cudaMemcpyAsync(host_norm[i],dev_norm[i],sizeof(double)*2*p[gpuid]*batch,cudaMemcpyDeviceToHost,stream[i]);
    }
    for(int i = 0;i < num_gpus;++i){
        for(int j = 0;j < 2*p;++j){
            host_rawnorm[i][host_order[i][j]] = host_norm[i][j];
        }
    }
    int* host_index=(int*)malloc(sizeof(int)*2*p[0]*num_gpus);
    fill_hostorder_total(raw_host_order,host_order,host_rawnorm,p,num_gpus);
    for(int i = 0;i < num_gpus*2*p[0];++i){
        host_index[raw_host_order[i]] = i;
    }
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(height >= 32){
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream1>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], height/32, height, width_perdevice, k);
        }
        else{
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream1>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], 1, height, width_perdevice, k);   
        }
        cudaDeviceSynchronize();
        computeFnorm2<<<batch, 32,0,stream[gpuid]>>>(dev_tempFnorm[gpuid], dev_Fnorm[gpuid], p[gpuid]);  //&1.3
    }
    double* Fin_Fnorm = (double*)malloc(sizeof(double)*batch);
    for(int bat = 0;bat < batch;++bat){
        for(int i = 0;i < num_gpus;++i)
            Fin_Fnorm[bat] += test_Fnorm[i][bat];
    }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(dev_Fnorm[gpuid],Fin_Fnorm,sizeof(double)*batch,cudaMemcpyHostToDevice,stream[gpuid]);
    }
    
    double* test_A = (double*)malloc(sizeof(double)*width_perdevice*height);
    printf("p %d p1 %d \n",p,p1);
    float elapsed_time = 0;
    float milliseconds = 0;
    int* host_order_total = (int*)malloc(sizeof(int)*2*p[0]*nums_gpu);
    while(!continue_flag){ 
        // part1
        dim3 dimGrid77(sliceNum, p[0], batch);// 2×2×100个block，每个block 256线程
        dim3 dimGrid7(p[0], batch, 1);
        // printf("EVD 1\n");
        // int times = 1;
        omp_set_num_threads(num_gpus);
        for(int i = 0;i < 2*p[0]-1;++i){
            #pragma omp parallel
            {
                int gpuid = omp_get_thread_num();
                cudaSetDevice(gpuid);
                generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
                generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
            }    
        }
        int init_time = 1;
        while(init_time <= nums_gpu){
            for(int total_time = 1;total_time < nums_gpu / init_time;++total_time){
                for(int i = 0;i<num_gpus;++i){
                    cudaSetDevice(i);
                    cudaMemcpyAsync(host_swap_data[i],dev_A[i]+p[i]*k*height,sizeof(double)*p[i]*k*height,cudaMemcpyDeviceToHost,stream[i]);
                }
                for(int per_time = 1;per_time <= init_time;++per_time){
                    for(int i = num_gpus/init_time * (per_time-1);i < num_gpus/init_time * per_time;++i){
                        cudaSetDevice(i);
                        if(i == nums_gpu/init_time * (per_time - 1)){
                            cudaMemcpyAsync(dev_A[i]+p[i]*k*height,host_swap_data[nums_gpu/init_time*(per_time-1)],sizeof(double)*p[i]*k*height,cudaMemcpyHostToDevice,stream[i]);
                        }
                        else
                            cudaMemcpyAsync(dev_A[i]+p[i]*k*height,host_swap_data[i+1],sizeof(double)*p[i]*k*height,cudaMemcpyHostToDevice,stream[i]);
                    }
                }
                for(int i = 0;i < p[0];++i){
                    #pragma omp parallel
                    {
                        int gpuid = omp_get_thread_num();
                        cudaSetDevice(gpuid);
                        generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
                        generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                        EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
                    }    
                }  
                
            }
            // refresh new part
            init_time *= 2;
            if(init_time <= nums_gpu){
                for(int i = 1;i <= init_time;++i){
                    int diff = i%2==1?p[0]*k*height:0;
                    for(int g = nums_gpu/init_time*(i-1);g < nums_gpu/init_time * i;++g){
                        cudaSetDevice(g);
                        cudaMemcpyAsync(host_swap_data[g],dev_A[g]+diff,sizeof(double)*p[g]*k*height,cudaMemcpyDeviceToHost,stream[g]);
                    }        
                } 
            }      
        }
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            // printf("gpuid %d \n",gpuid);
            compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width_perdevice, p[gpuid], q, k);
            binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);
            
        } 
        for(int i = 0;i < num_gpus;++i){
            cudaSetDevice(i);
            cudaMemcpyAsync(host_order[i],dev_order[i],sizeof(int)*2*p[gpuid]*batch,cudaMemcpyDeviceToHost,stream[i]);
            cudaMemcpyAsync(host_norm[i],dev_norm[i],sizeof(double)*2*p[gpuid]*batch,cudaMemcpyDeviceToHost,stream[i]);
        }
        for(int i = 0;i < num_gpus;++i){
            for(int j = 0;j < 2*p;++j){
                host_rawnorm[i][host_order[i][j]] = host_norm[i][j];
            }
        }

        fill_hostorder_total(host_order_total,host_order,host_rawnorm,p,num_gpus);
        for(int i = 0;i < nums_gpu;++i){
            cudaSetDevice(i);
            cudaMemcpyAsync(host_A+i*width_perdevice*height,dev_A[i],sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream[i]);
        }
        for(int i = 0;i<nums_gpu;++i){
            cudaSetDevice(i);
            int cnt = 0;
            for(int index = i*2*p[0];index < (i+1)*2*p[0];++index){
                cudaMemcpyAsync(dev_A[i]+cnt*k*height,&host_A[host_order_total[host_index[i]]*k*height],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream[i]);
                ++cnt;
            }
        }
        ++sweep;
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            judgeFunc<<<batch, 1024,0,stream[gpuid]>>>(dev_allpass[gpuid], dev_pass[gpuid], p[gpuid]);   // concentrate each block's result(converged or not)
        }
        bool tempFlag = true;
        for(int i = 0;i < nums_gpu;++i){
            if(!ifallpass(host_allpass[i],batch,p[i])){
                tempFlag = false;
            }
        }
        continue_flag = (tempFlag || sweep>maxsweep);
    }
        
#pragma region       
        // for(int gpu_index = 0;gpu_index<num_gpus-1;++gpu_index){
        //     for(int i = 0;i<num_gpus;++i){
        //         cudaSetDevice(i);
        //         cudaMemcpyAsync(host_swap_data[i],dev_A[i]+p[gpuid]*k*height,sizeof(double)*p[gpuid]*k*height,cudaMemcpyDeviceToHost,stream[gpuid]);
        //     }
        //     for(int i = 0;i < num_gpus;++i){
        //         cudaSetDevice(gpuid);
        //         if(i == num_gpus-1){
        //             cudaMemcpyAsync(dev_A[gpuid]+p[gpuid]*k*height,host_swap_data[0],sizeof(double)*p[gpuid]*k*height,cudaMemcpyHostToDevice,stream[gpuid]);
        //         }
        //         else{
        //             cudaMemcpyAsync(dev_A[gpuid]+p[gpuid]*k*height,host_swap_data[gpuid+1],sizeof(double)*p[gpuid]*k*height,cudaMemcpyHostToDevice,stream[gpuid]);
        //         }
        //     }
        //     // part2
        //     for(int i = 0;i < p;++i){
        //         #pragma omp parallel
        //         {
        //             int gpuid = omp_get_thread_num();
        //             cudaSetDevice(gpuid);
        //             generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
        //             generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
        //             EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
        //         }    
        //     }  
        // }

        // for(int gpu_index = 0;gpu_index<num_gpus-1;++gpu_index){
        //     for(int i = 0;i<num_gpus;++i){
        //         cudaSetDevice(i);
        //         if(i != 0)
        //             cudaMemcpyAsync(host_swap_data[i],dev_A[i],sizeof(double)*p[gpuid]*k*height,cudaMemcpyDeviceToHost,stream[gpuid]);
        //         else{
        //             cudaMemcpyAsync(host_swap_data[i],dev_A[i]+p*k*height,sizeof(double)*p[gpuid]*k*height,cudaMemcpyDeviceToHost,stream[gpuid]);
        //         }
        //     }
        //     for(int i = 0;i < num_gpus;++i){
        //         cudaSetDevice(gpuid);
        //         if(i == num_gpus-1){
        //             cudaMemcpyAsync(dev_A[gpuid]+p[gpuid]*k*height,host_swap_data[0],sizeof(double)*p[gpuid]*k*height,cudaMemcpyHostToDevice,stream[gpuid]);
        //         }
        //         else{
        //             cudaMemcpyAsync(dev_A[gpuid]+p[gpuid]*k*height,host_swap_data[gpuid+1],sizeof(double)*p[gpuid]*k*height,cudaMemcpyHostToDevice,stream[gpuid]);
        //         }
        //     }
        //     // part2
        //     for(int i = 0;i < p;++i){
        //         #pragma omp parallel
        //         {
        //             int gpuid = omp_get_thread_num();
        //             cudaSetDevice(gpuid);
        //             generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
        //             generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
        //             EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
        //         }    
        //     }  
        // }
        // // part3
        // cudaSetDevice(gpu0);
        // cudaMemcpyAsync(host_swap_data,dev_A+p*k*height,sizeof(double)*p*k*height,cudaMemcpyDeviceToHost,stream1);
        // cudaSetDevice(gpu1);
        // cudaMemcpyAsync(host_swap_data_1,dev_A_1,sizeof(double)*p1*k*height,cudaMemcpyDeviceToHost,stream2);

        // cudaSetDevice(gpu0);
        // cudaMemcpyAsync(dev_A+p*k*height,host_swap_data_1,sizeof(double)*p*k*height,cudaMemcpyHostToDevice,stream1);
        // cudaSetDevice(gpu1);
        // cudaMemcpyAsync(dev_A_1,host_swap_data,sizeof(double)*p1*k*height,cudaMemcpyHostToDevice,stream2);


        // for(int i = 0;i < p;++i){
        //     #pragma omp parallel
        //     {
        //         int gpuid = omp_get_thread_num();
        //         if(gpuid == 0){
        //             cudaSetDevice(gpu0);
        //             generate_jointG00_1<<<dimGrid77, 256,0,stream1>>>(dev_pab1,dev_A, height, width_perdevice, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj,i,  k, slice, sliceNum);    //&1.3
        //             generate_jointG21<<<dimGrid7, 256,0,stream1>>>(dev_jointG, dev_AiAi, dev_AiAj, dev_AjAj, dev_Fnorm, dev_pass, p, k, sliceNum, svd_tol);    //&1.3
        //             EVD_1(stream1,dev_jointG, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width_perdevice, dev_roundRobin, batch, k, slice, sliceNum, sweep); //&1.3
        //         }
        //         else{
        //             cudaSetDevice(gpu1);
        //             generate_jointG00_1<<<dimGrid77, 256,0,stream2>>>(dev_pab1_1,dev_A_1, height, width_perdevice, p1, q, dev_pairsOfEVD_1, dev_AiAi_1, dev_AiAj_1, dev_AjAj_1,i,  k, slice, sliceNum);    //&1.3
        //             generate_jointG21<<<dimGrid7, 256,0,stream2>>>(dev_jointG_1, dev_AiAi_1, dev_AiAj_1, dev_AjAj_1, dev_Fnorm_1, dev_pass_1, p1, k, sliceNum, svd_tol);    //&1.3
        //             EVD_1(stream2,dev_jointG_1, dev_A_1, dev_V_1, dev_pairsOfEVD_1, p1, q, height, width_perdevice, dev_roundRobin_1, batch, k, slice, sliceNum, sweep); //&1.3
        //         }

        //     }     
        // }

        // #pragma omp parallel
        // {
        //     int gpuid = omp_get_thread_num();
            
        //     // printf("gpuid %d \n",gpuid);
        //     if(gpuid==0){
        //         cudaSetDevice(gpu0);
        //         compute_norm<<<2 * p * batch, 128,0,stream1>>>(dev_A, dev_norm, dev_order, height, width_perdevice, p, q, k);
        //         binoticSort_original<<<batch, 1024,0,stream1>>>(dev_norm, dev_order, 2 * p, p);
        //     }
        //     else{
        //         cudaSetDevice(gpu1);
        //         compute_norm<<<2 * p1 * batch, 128,0,stream2>>>(dev_A_1, dev_norm_1, dev_order_1, height, width_perdevice, p1, q, k);
        //         binoticSort_original<<<batch, 1024,0,stream2>>>(dev_norm_1, dev_order_1, 2 * p1, p1);
        //     }
        // } 
        // cudaStreamSynchronize(stream1);
        // cudaMemcpyAsync(host_order_1,dev_order,sizeof(int)*2*p*batch,cudaMemcpyDeviceToHost,stream1);
        // cudaMemcpyAsync(host_norm_1,dev_norm,sizeof(double)*2*p*batch,cudaMemcpyDeviceToHost,stream1);
        // cudaStreamSynchronize(stream2);
        // cudaMemcpyAsync(host_order_2,dev_order_1,sizeof(int)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
        // cudaMemcpyAsync(host_norm_2,dev_norm_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);

        // cudaStreamSynchronize(stream1);
        // cudaStreamSynchronize(stream2);

        // #pragma omp parallel for
        // for(int i = 0;i < 2*p;++i){
        //     // printf("%d ",host_order_1[i]);
        //     host_rawnorm_1[host_order_1[i]] = host_norm_1[i];
        // }
        // #pragma omp parallel for
        // for(int i = 0;i < 2*p1;++i){
        //     // printf("%d ",host_order_2[i]);
        //     host_rawnorm_2[host_order_2[i]] = host_norm_2[i];
        // }
        // fill_hostorder_total(host_order_total,host_order_1,host_order_2,host_rawnorm_1,host_rawnorm_2,p,p1);
        // cudaMemcpyAsync(host_A,dev_A,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream1);
        // cudaMemcpyAsync(host_A+width_perdevice*height,dev_A_1,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream2);
        // for(int i = 0;i < 2*p;++i){
        //     cudaMemcpyAsync(dev_A+i*k*height,&host_A[host_order_total[host_index[i]]*k*height],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream1);
        // }

        // for(int i = 0;i < 2*p1;++i){
        //     cudaMemcpyAsync(dev_A_1+i*k*height,host_A+host_order_total[host_index[i+2*p]]*k*height,sizeof(double)*k*height,cudaMemcpyHostToDevice,stream2);
        // }
       
        // ++sweep;
        // cudaSetDevice(gpu0);
        // judgeFunc<<<batch, 1024,0,stream1>>>(dev_allpass, dev_pass, p);   // concentrate each block's result(converged or not)
        // cudaSetDevice(gpu1);
        // judgeFunc<<<batch, 1024,0,stream2>>>(dev_allpass_1, dev_pass_1, p1);   // concentrate each block's result(converged or not)
        // cudaMemcpyAsync(host_allpass, dev_allpass, sizeof(unsigned) * batch, cudaMemcpyDeviceToHost,stream1);    
        // cudaMemcpyAsync(host_allpass_1, dev_allpass_1, sizeof(unsigned) * batch, cudaMemcpyDeviceToHost,stream2);
        // printf("host pass\n");
        // for(int i = 0;i < batch;++i){
        //     printf("%d ",host_allpass[i]);
        // }
        // for(int i = 0;i < batch;++i){
        //     printf("%d ",host_allpass_1[i]);
        // }
        // printf("\n");

        // continue_flag = ((ifallpass(host_allpass, batch, p) && ifallpass(host_allpass_1,batch,p1)) || sweep>maxsweep);
        // cudaSetDevice(gpu0);
        
        // break;
#pragma endregion
    
    // cudaEventRecord(stop,stream1);
    // cudaEventSynchronize(stop);
    
    // // 6️⃣ 计算时间（毫秒）
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // elapsed_time += milliseconds/1000;
    // printf("it costs %f \n",elapsed_time);
    dim3 dimGrid10(2 * p, batch, 1);
    dim3 dimBlock10(32, k, 1);
    #pragma omp parallel
    {
        cudaSetDevice(gpuid);
        int gpuid = omp_get_thread_num();
        getUDV<<<dimGrid10, dimBlock10,0,stream[gpuid]>>>(dev_A[gpuid], dev_U[gpuid], dev_V[gpuid], dev_V0[gpuid], height, width_perdevice, height, width_perdevice, p[gpuid], height/32, dev_diag[gpuid], width_perdevice, k);  //&1.3
        
    } 
    printf("sweep:%d \n",sweep);

    // double* host_diag = (double*)malloc(sizeof(double)*minmn*batch);
    // double* host_diag1 = (double*)malloc(sizeof(double)*minmn*batch);
    // cudaSetDevice(gpu0);
    // cudaMemcpy(host_diag,dev_diag,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    // cudaSetDevice(gpu0);
    // cudaMemcpy(host_diag1,dev_diag_1,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    // FILE* file2 = fopen("dev_diag.txt","w");
    // for(int f = 0;f < minmn;++f){
    //     fprintf(file2,"%lf %lf ",host_diag[f],host_diag1[f]);
    // }
    // double* host_U = (double*)malloc(sizeof(double)*height*height*batch);
    // cudaMemcpy(host_U,dev_U,sizeof(double)*height*height*batch,cudaMemcpyDeviceToHost);
    // FILE* file = fopen("dev_U.txt","w");
    // for(int f = 0;f < height;++f){
    //     for(int g=0;g<height;++g){
    //         fprintf(file,"%lf ",host_U[f*height+g]);
    //     }
    //     fprintf(file,"\n");
    // }
    
    // printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

    free(host_A);
    // cudaFree(dev_A);
    // cudaFree(dev_U);
    // cudaFree(dev_V);
    // cudaFree(dev_diag);
    cudaDeviceReset();
}

int main(int argc, char* argv[]){
    test17();
}