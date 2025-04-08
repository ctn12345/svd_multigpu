#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include<omp.h>
// #include <nvToolsExt.h>
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
        double tmp = DBL_MAX;
        int need_index = 0;
        for(int u = 0;u < num_gpus;++u){
            if(idx[u] < 2 * p[u] && tmp > host_norm[u][host_order[u][idx[u]]]){
                tmp = host_norm[u][host_order[u][idx[u]]];
                need_index = u;
            }
        }
        
        host_order_total[index] = host_order[need_index][idx[need_index]]+p[0]*2*need_index;
        idx[need_index]++;
        index++;
        flag = true;
        for(int u = 0;u < num_gpus;++u){
            if(idx[u] < 2*p[u]){
                flag = false;
            }
        }
    }
    free(idx);
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
    printf("nums gpu is %d\n ",num_gpus);
    int gpu0=0,gpu1=1;
    int batch = 1;
    int height = 64;
    int width = 64;
    int th=0, tw=0;
    // int shape[3] = {batch, height, width};
    int minmn = height > width/num_gpus ? width/num_gpus : height;

    double* host_A = (double*)malloc(sizeof(double) * height * width);
    double* host_V = (double*)malloc(sizeof(double) * width*width);
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
    double** dev_A = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_V = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_V0 = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_U = (double**)malloc(num_gpus * sizeof(double*));
    int** dev_roundRobin = (int**)malloc(num_gpus * sizeof(int*));
    double** dev_jointG = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_AiAi= (double**)malloc(num_gpus * sizeof(double*));
    double** dev_Aij = (double**)malloc(num_gpus * sizeof(double*)); 
    double** dev_AiAj = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_AjAj = (double**)malloc(num_gpus * sizeof(double*));
    unsigned** dev_allpass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    unsigned** dev_pass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    double** dev_swap_data = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_norm = (double**)malloc(num_gpus * sizeof(double*));
    unsigned int** dev_order= (unsigned int**)malloc(num_gpus * sizeof(unsigned int*));
    double** dev_tempFnorm= (double**)malloc(num_gpus * sizeof(double*));
    double** dev_Fnorm= (double**)malloc(num_gpus * sizeof(double*));
    double** dev_diag = (double**)malloc(num_gpus * sizeof(double*));
    
    int** dev_pairsOfEVD = (int**)malloc(num_gpus * sizeof(int*));

    int* p = (int*)malloc(num_gpus * sizeof(int));
    int q;
    int sliceNum;
   
    unsigned** host_allpass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    unsigned** host_pass = (unsigned**)malloc(num_gpus * sizeof(unsigned*));
    double** value = (double**)malloc(num_gpus * sizeof(double*));
    double** host_Fnorm = (double**)malloc(num_gpus * sizeof(double*));
    double** host_A_per = (double**)malloc(num_gpus * sizeof(double*));
    int** host_order = (int**)malloc(num_gpus * sizeof(int*));
    double** host_rawnorm = (double**)malloc(num_gpus*sizeof(double*));
    double** host_norm = (double**)malloc(num_gpus*sizeof(double*));
    double** host_swap_data = (double**)malloc(sizeof(double*)*num_gpus);
    double** host_swap_V = (double**)malloc(sizeof(double*)*num_gpus);
    double** test_Fnorm = (double**)malloc(sizeof(double*)*num_gpus);
    // int gpuid = 0;
    for(int gpuid = 0;gpuid < num_gpus;++gpuid){
        p[gpuid]=(width_perdevice-1)/(2*k)+1;
        host_order[gpuid] = (int*)malloc(sizeof(int)*p[gpuid]*2*batch);
        host_A_per[gpuid] = (double*)malloc(sizeof(double)*width_perdevice*height);
        
        host_rawnorm[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*2*batch);
        host_norm[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*2*batch);
        host_swap_data[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*height*k);
        host_swap_V[gpuid] = (double*)malloc(sizeof(double)*width_perdevice*width/2);
        test_Fnorm[gpuid] = (double*)malloc(sizeof(double)*batch);
        host_pass[gpuid] = (unsigned*)malloc(sizeof(unsigned)*p[gpuid]*2*batch);
        host_allpass[gpuid] = (unsigned*)malloc(sizeof(unsigned)*batch);
    }
    q = (height-1)/slice+1;
    sliceNum = q;
#pragma endregion
cudaError_t err;
for(int gpuid = 0;gpuid < num_gpus;++gpuid){
    cudaSetDevice(gpuid);
        
    err = cudaMalloc((void**)&dev_pa[gpuid],sizeof(int)*p[gpuid]);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMalloc((void**)&dev_pb[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab[gpuid],sizeof(int)*2*p[gpuid]*(2*p[gpuid]-1));
    // next_time
    cudaMalloc((void**)&dev_pa1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pb1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab1[gpuid],sizeof(int)*2*p[gpuid]*p[gpuid]);

    cudaMalloc((void **)&dev_U[gpuid], sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_A[gpuid], sizeof(double) * height * width_perdevice * batch);
    err = cudaMalloc((void **)&dev_V[gpuid], sizeof(double) * width * width_perdevice * batch);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMalloc((void **)&dev_V0[gpuid], sizeof(double) * width * width_perdevice * batch);
    cudaMalloc((void **)&dev_diag[gpuid],sizeof(double) * minmn);
    cudaMalloc((void **)&dev_roundRobin[gpuid], sizeof(int) * (2 * k - 1) * 2 * k);
    cudaMalloc((void **)&dev_jointG[gpuid], sizeof(double) * 2*k * 2*k * p[gpuid]*batch);
    cudaMalloc((void **)&dev_Aij[gpuid], sizeof(double) * height * 2*k * p[gpuid]*batch);
    cudaMalloc((void **)&dev_AiAi[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_AiAj[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_AjAj[gpuid], sizeof(double) * k * k * sliceNum * p[gpuid] * batch);
    cudaMalloc((void **)&dev_pairsOfEVD[gpuid], sizeof(int) * 2 * p[gpuid] * batch);
    cudaMalloc((void **)&dev_swap_data[gpuid],sizeof(double)*p[gpuid]*height*k);
    cudaMalloc((void **)&dev_pass[gpuid],sizeof(unsigned)*p[gpuid]*batch);
    cudaMalloc((void **)&dev_norm[gpuid],sizeof(double)*2*p[gpuid]*batch);
    cudaMalloc((void **)&dev_Fnorm[gpuid], sizeof(double) * batch);
    cudaMalloc((void **)&dev_tempFnorm[gpuid], sizeof(double) * 2*p[gpuid]*batch);
    cudaMalloc((void **)&dev_order[gpuid],sizeof(unsigned int)*2*p[gpuid]*batch);
    cudaMalloc((void **)&dev_allpass[gpuid],sizeof(unsigned)*batch);
    err = cudaMemset(dev_V[gpuid], 0,  sizeof(double)*width * width_perdevice * batch);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(dev_U[gpuid], 0,  sizeof(double)*height * height * batch);
    cudaMemset(dev_diag[gpuid], 0,  sizeof(double)*minmn);
    cudaMemset(dev_V0[gpuid], 0, sizeof(double) * width * width_perdevice * batch);
    cudaMemset(dev_pairsOfEVD[gpuid], 0,  sizeof(int) *2 * p[gpuid] * batch); 
    cudaMemset(dev_pass[gpuid], 0,  sizeof(unsigned)*p[gpuid] * batch);
}
    int shape[3]={batch,height,width_perdevice};
    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 2.0;

    cudaStream_t* stream = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(dev_A[i],host_A+i*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyHostToDevice,stream[i]);
    }
    omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        Multi_init_dev_V<<<batch,256,0,stream[gpuid]>>>(dev_V[gpuid],width_perdevice,width,gpuid);
    }
    clock_t c1 = clock();
    dim3 dimGrid0(1, 1, 1);
    dim3 dimBlock0(32, 32, 1);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream[gpuid]>>>(dev_roundRobin[gpuid], 2*k);
    }
    
    // printf("dev_V \n");
    // cudaSetDevice(0);
    // double* test_V = (double*)malloc(sizeof(double)*width*width_perdevice);
    // cudaMemcpyAsync(test_V,dev_V[0],sizeof(double)*width*width_perdevice,cudaMemcpyDeviceToHost,stream[0]);
    // for(int i = 0;i < width_perdevice;++i){
    //    for(int g = 0;g < width;++g){
    //        printf("%f ",test_V[i*width+g]);
    //    }
    //    printf("\n");
    // }
    // printf("dev_1\n");
    // cudaSetDevice(1);
    // cudaMemcpyAsync(test_V,dev_V[1],sizeof(double)*width*width_perdevice,cudaMemcpyDeviceToHost,stream[1]);
    // for(int i = 0;i < width_perdevice;++i){
    //    for(int g = 0;g < width;++g){
    //     if(test_V[i*width+g] == 1){
    //         printf("%d %d ",i,g);
    //         printf("\n");
    //     }
        
    //     //    printf("%f ",test_V[i*width+g]);
    //    }
    //    printf("\n");
    // }
    bool continue_flag = false;

    // omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        getRankNewNew_2<<<1,1024,0,stream[gpuid]>>>(2*p[gpuid],dev_pab[gpuid],dev_pa[gpuid],dev_pb[gpuid]);
        getRankNewNew_1<<<1,1024,0,stream[gpuid]>>>(p[gpuid],dev_pa1[gpuid],dev_pb1[gpuid],dev_pab1[gpuid]);
    }
    // printf("pab %d \n",p[0]);

    // int* test_pab=(int*)malloc(sizeof(int)*2*p[0]*(2*p[0]-1));
    // cudaMemcpyAsync(test_pab,dev_pab[0],sizeof(int)*(2*p[0]-1)*2*p[0],cudaMemcpyDeviceToHost,stream[0]);
    // for(int i = 0;i < (2*p[0]-1)*2*p[0];++i){
    //     printf("%d ",test_pab[i]);
    // }
    // printf("\n");
    int sweep = 0,maxsweep = 11;
    double svd_tol = 1e-7;
    int* raw_host_order = (int*)malloc(sizeof(int)*2*p[0]*num_gpus*batch);
    cudaStreamSynchronize(stream[0]);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width_perdevice, p[gpuid], q, k);
        binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);
        
    } 
    // cudaDeviceSynchronize();
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(host_order[i],dev_order[i],sizeof(int)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
        cudaMemcpyAsync(host_norm[i],dev_norm[i],sizeof(double)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
    }
    for(int i = 0;i < num_gpus;++i){
        for(int j = 0;j < 2*p[0];++j){
            host_rawnorm[i][host_order[i][j]] = host_norm[i][j];
        }
    }
    // printf("host norm \n");
    // for(int g = 0;g < 2*p[0];++g){
    //     printf("%f ",host_norm[0][g]);
    // }
    // printf("\n");
    int* host_index=(int*)malloc(sizeof(int)*2*p[0]*num_gpus);
    // int* host_pass = (int*)malloc(sizeof(int)*10);
    fill_hostorder_total(raw_host_order,host_order,host_rawnorm,p,num_gpus);
    // test raw_host_order part
    // for(int g = 0;g < num_gpus*2*p[0];++g){
    //     int gpuid = raw_host_order[g] / (2*p[0]);
    //     printf("%f ",host_rawnorm[gpuid][raw_host_order[g]-gpuid * 2*p[0]]);
    // }
    // printf("\n");
    for(int i = 0;i < num_gpus*2*p[0];++i){
        host_index[raw_host_order[i]] = i;
    }
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(height >= 32){
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], height/32, height, width_perdevice, k);
        }
        else{
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], 1, height, width_perdevice, k);   
        }
        // cudaDeviceSynchronize();
        computeFnorm2<<<batch, 32,0,stream[gpuid]>>>(dev_tempFnorm[gpuid], dev_Fnorm[gpuid], p[gpuid]);  //&1.3
    }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpy(test_Fnorm[i],dev_Fnorm[i],sizeof(double)*batch,cudaMemcpyDeviceToHost);
    }
    double* Fin_Fnorm = (double*)malloc(sizeof(double)*batch);
    for(int bat = 0;bat < batch;++bat){
        for(int i = 0;i < num_gpus;++i)
            Fin_Fnorm[bat] += test_Fnorm[i][bat];
    }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(dev_Fnorm[i],Fin_Fnorm,sizeof(double)*batch,cudaMemcpyHostToDevice,stream[i]);
    }
    // before have been tested

    float elapsed_time = 0;
    float milliseconds = 0;
    int* host_order_total = (int*)malloc(sizeof(int)*2*p[0]*num_gpus);
    printf("FIN Form %f \n",Fin_Fnorm[0]);
    double* host_jointG = (double*)malloc(sizeof(double)*2*k*2*k);
    double* test_A = (double*)malloc(sizeof(double)*p[0]*k*height);
    double* test_aij = (double*)malloc(sizeof(double)*k);
    cudaError_t errf;
    while(!continue_flag){ 
        // part1
        dim3 dimGrid77(sliceNum, p[0], batch);// 2×2×100个block，每个block 256线程
        dim3 dimGrid7(p[0], batch, 1);
        // printf("EVD 1\n");
        // int times = 1;
        omp_set_num_threads(num_gpus);
        
        // cudaMemcpy(test_A,dev_A[0],sizeof(double)*p[0]*k*height,cudaMemcpyDeviceToHost);
        // printf("devA\n");
        // for(int g = 0;g < p[0];++g){
        //     for(int f = 0;f < 5;++f)
        //         printf("%f ",test_A[g*k*height + f]);
        //     printf("\n");
        // }
        // printf("\n");
       
        // printf("EVD start \n");
        for(int i = 0;i < 2*p[0]-1;++i){
            #pragma omp parallel
            {
                int gpuid = omp_get_thread_num();
                cudaSetDevice(gpuid);
                generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
               
                generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height,width, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
            }    
        }
        //test 1
        // printf("\nEVD1\n");
        
        // cudaMemcpy(host_jointG,dev_jointG[0],sizeof(double)*4*k*k,cudaMemcpyDeviceToHost);
        // for(int i = 0;i < 10;++i){
        //     printf("%f ",host_jointG[i]);
        // }
        // printf("\n");

        // break;

        int init_time = 1;
        while(init_time < num_gpus){
            for(int total_time = 1;total_time < num_gpus / init_time;++total_time){
                for(int i = 0;i<num_gpus;++i){
                    cudaSetDevice(i);
                    cudaMemcpyAsync(host_swap_data[i],dev_A[i]+p[i]*k*height,sizeof(double)*p[i]*k*height,cudaMemcpyDeviceToHost,stream[i]);
                }
                for(int per_time = 1;per_time <= init_time;++per_time){
                    for(int i = num_gpus/init_time * (per_time-1);i < num_gpus/init_time * per_time;++i){
                        cudaSetDevice(i);
                        if(i == num_gpus/init_time * per_time-1){
                            cudaMemcpyAsync(dev_A[i]+p[i]*k*height,host_swap_data[num_gpus/init_time*(per_time-1)],sizeof(double)*p[i]*k*height,cudaMemcpyHostToDevice,stream[i]);
                            cudaMemcpyAsync(dev_V[i]+width*width_perdevice/2,host_swap_V[num_gpus/init_time*(per_time-1)],sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[i]);
                        }
                        else{
                            cudaMemcpyAsync(dev_A[i]+p[i]*k*height,host_swap_data[i+1],sizeof(double)*p[i]*k*height,cudaMemcpyHostToDevice,stream[i]);
                            cudaMemcpyAsync(dev_V[i]+width*width_perdevice/2,host_swap_V[i+1],sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[i]);
                        }    
                    }
                }
                for(int i = 0;i < p[0];++i){
                    #pragma omp parallel
                    {
                        int gpuid = omp_get_thread_num();
                        cudaSetDevice(gpuid);
                        generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
                        generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                        MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width,width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
                    }    
                }  
                
            }
            // refresh new part
            init_time *= 2;
            if(init_time <= num_gpus){
                for(int i = 1;i <= init_time;++i){
                    int diff = i%2==1?p[0]*k*height:0;
                    int diff_V = i%2 == 1?width_perdevice*width/2:0;
                    for(int g = num_gpus/init_time*(i-1);g < num_gpus/init_time * i;++g){
                        cudaSetDevice(g);
                        cudaMemcpyAsync(host_swap_data[g],dev_A[g]+diff,sizeof(double)*p[g]*k*height,cudaMemcpyDeviceToHost,stream[g]);
                        cudaMemcpyAsync(host_swap_V[g],dev_V[g]+diff_V,sizeof(double)*width_perdevice*width/2,cudaMemcpyDeviceToHost,stream[g]);
                    }        
                }
                for(int i = 1;i <= init_time;++i){
                    int diff = i%2==1?p[0]*k*height:0;
                    int diff_V = i%2 == 1?width_perdevice*width/2:0;
                    int flag = i%2;
                    for(int g = num_gpus/init_time*(i-1);g < num_gpus/init_time * i;++g){
                        cudaSetDevice(g);
                        if(flag == 0){
                            cudaMemcpyAsync(dev_A[g]+diff,host_swap_data[g-num_gpus/init_time],sizeof(double)*p[g]*k*height,cudaMemcpyHostToDevice,stream[g]);
                            cudaMemcpyAsync(dev_V[i]+diff_V,host_swap_V[g-num_gpus/init_time],sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[g]);
                        }
                        else{
                            cudaMemcpyAsync(dev_A[g]+diff,host_swap_data[g+num_gpus/init_time],sizeof(double)*p[g]*k*height,cudaMemcpyHostToDevice,stream[g]);
                            cudaMemcpyAsync(dev_V[i]+diff_V,host_swap_V[g+num_gpus/init_time],sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[g]);
                        }
                        }
                           
                    }        
                
                for(int i = 0;i < p[0];++i){
                    #pragma omp parallel
                    {
                        int gpuid = omp_get_thread_num();
                        cudaSetDevice(gpuid);
                        generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,  k, slice, sliceNum);    //&1.3
                        generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                        MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width,width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
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
            cudaMemcpyAsync(host_order[i],dev_order[i],sizeof(int)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
            cudaMemcpyAsync(host_norm[i],dev_norm[i],sizeof(double)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
        }
        for(int i = 0;i < num_gpus;++i){
            for(int j = 0;j < 2*p[0];++j){
                host_rawnorm[i][host_order[i][j]] = host_norm[i][j];
            }
        }

        fill_hostorder_total(host_order_total,host_order,host_rawnorm,p,num_gpus);
        // printf("host order total\n");
        for(int g = 0;g < num_gpus*p[0]*2;++g){
            // printf("%d ",host_order_total[g]);
            int gpuid = host_order_total[g] / (2*p[0]);
            int index = host_order_total[g] % (2*p[0]);
            // printf("%f ",host_rawnorm[gpuid][index]);
        }
        // printf("\n");
        // cudaMemcpy(host_pass[0],dev_pass[0],sizeof(int)*p[0],cudaMemcpyDeviceToHost);
        // printf("host pass\n");
        // for(int g = 0;g < p[0];++g){
        //     printf("%d ",host_pass[0][0]);
        // }
        // printf("\n");
        for(int i = 0;i < num_gpus;++i){
            cudaSetDevice(i);
            cudaMemcpyAsync(host_A+i*width_perdevice*height,dev_A[i],sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream[i]);
            cudaMemcpyAsync(host_V+i*width_perdevice*width,dev_V[i],sizeof(double)*width_perdevice*width,cudaMemcpyDeviceToHost,stream[i]);
        }
        int cnt = 0;
        for(int i = 0;i<num_gpus;++i){
            cudaSetDevice(i);
            int per_len = 0;
            for(int index = i*2*p[0];index < (i+1)*2*p[0];++index){
                cudaMemcpyAsync(dev_A[i]+per_len*k*height,&host_A[host_order_total[host_index[cnt]]*k*height],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream[i]);
                cudaMemcpyAsync(dev_V[i]+per_len*k*width,&host_V[host_order_total[host_index[cnt]]*k*width],sizeof(double)*k*width,cudaMemcpyHostToDevice,stream[i]);
                ++cnt;
                per_len++;
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
        for(int i = 0;i < num_gpus;++i){
            cudaMemcpy(host_allpass[i],dev_allpass[i],sizeof(unsigned)*batch,cudaMemcpyDeviceToHost);
        }
        // printf("host pass\n");
        for(int i = 0;i < num_gpus;++i){
            // printf("%d ",host_allpass[i][0]);
            if(!ifallpass(host_allpass[i],batch,p[i])){
                tempFlag = false;
            }
        }
        // printf("\n");
        continue_flag = (tempFlag || sweep>maxsweep);
        for(int g = 0;g < num_gpus;++g){
            cudaSetDevice(g);
            cudaStreamSynchronize(stream[g]);
        }
    }
     #pragma omp parallel
     {
        //  int gpuid = omp_get_thread_num();
         
         int gpuid = omp_get_thread_num();
         cudaSetDevice(gpuid);
         

         dim3 dimGrid10(2 * p[gpuid], batch, 1);
         dim3 dimBlock10(32, k, 1);
         getUDV<<<dimGrid10, dimBlock10,0,stream[gpuid]>>>(dev_A[gpuid], dev_U[gpuid], dev_V[gpuid], dev_V0[gpuid], height, width_perdevice, height, width_perdevice, p[gpuid], height/32, dev_diag[gpuid], width_perdevice, k);  //&1.3
        //  gpuid = 1;
        //  cudaSetDevice(gpuid);
        //  // int gpuid = omp_get_thread_num();
        //  getUDV<<<dimGrid10, dimBlock10,0,stream[gpuid]>>>(dev_A[gpuid], dev_U[gpuid], dev_V[gpuid], dev_V0[gpuid], height, width_perdevice, height, width_perdevice, p[gpuid], height/32, dev_diag[gpuid], width_perdevice, k);  //&1.3
     }
     clock_t c2 = clock();
     printf("it costs %f s\n",(double)(c2-c1)/CLOCKS_PER_SEC); 
    //  cudaError_t err1 = cudaGetLastError();
    // if (err1 != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err1));
    // }
     printf("sweep:%d \n",sweep);
     double** host_diag = (double**)malloc(sizeof(double*)*num_gpus);
     for(int i = 0;i < num_gpus;++i){
         host_diag[i] = (double*)malloc(sizeof(double)*minmn*batch);
     }
     FILE* file2 = fopen("dev_diag.txt","w");
    for(int i = 0;i < num_gpus;++i){
        cudaMemcpy(host_diag[i],dev_diag[i],sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
     }
     for(int i = 0;i < num_gpus;++i){
         for(int g = 0;g<minmn;++g)
         fprintf(file2,"%lf ",host_diag[i][g]);
     }

     for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpyAsync(host_V+i*width*width_perdevice,dev_V[i],sizeof(double)*width_perdevice*width,cudaMemcpyDeviceToHost,stream[i]);
     }
    FILE* file_V = fopen("dev_V.txt","w");
    for(int i  = 0;i < width;++i){
        for(int j =0;j < width;++j){
            fprintf(file_V,"%lf ",host_V[i*width+j]);
        }
        fprintf(file_V,"\n");
    }
    // for(int i = 0;i < num_gpus;++i){
    //     cudaSetDevice(i);
    //     cudaFree(dev_A[i]);
    //     cudaFree(dev_U[i]);
    //     cudaFree(dev_V[i]);
    //     cudaFree(dev_V0[i]);
    //     cudaStreamDestroy(stream[i]);
    // }
    // free(host_A);
}

int main(int argc, char* argv[]){
    test17();
    // cublasDestroy(handle);
    // cudaDeviceReset();
}