#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include<omp.h>
#include <nvToolsExt.h>
#include "matrix_generate.hpp"
#include "large_matrix_svd_kernels.cu"
#include "small_matrix_svd.cu"
using namespace std;
using namespace std::chrono;
void fill_hostorder_total_new(int* host_order_total, unsigned int* host_order, double* host_norm, int* p, int num_gpus, int batch) {
    int** idx = (int**)malloc(sizeof(int*) * batch);
    for (int i = 0; i < batch; ++i) {
        idx[i] = (int*)malloc(sizeof(int) * num_gpus);
        memset(idx[i], 0, sizeof(int) * num_gpus);
    }

    int* need_index = (int*)malloc(sizeof(int) * batch);
    memset(need_index, 0, sizeof(int) * batch);

    bool flag = false;
    int index = 0;
    // printf("begin\n");
    for (int number = 0; number < batch; ++number) {
        flag = false;
        while(!flag){
            double tmp = DBL_MAX;
            for (int u = 0; u < num_gpus; ++u) {
                int diff_index = u * batch * 2 * p[u]+number * 2 * p[u];
                if (idx[number][u] < 2 * p[u] &&
                    tmp > host_norm[diff_index + host_order[diff_index + idx[number][u]]]) {
                    
                    tmp = host_norm[diff_index + host_order[diff_index + idx[number][u]]];
                    need_index[number] = u;
                }
            }
            // printf("tmp %f \n",tmp);
            int cur_gpu = need_index[number];
            int cur_order_idx = 2 * p[0] * batch * cur_gpu;

            int cur_offset = idx[number][cur_gpu] + cur_order_idx;

            host_order_total[index] = host_order[cur_offset] + p[0] * 2 * cur_gpu;
               
            idx[number][cur_gpu]++;
            index++;
            flag = true;
            for (int u = 0; u < num_gpus; ++u) {
                if (idx[number][u] < 2 * p[u]) {
                    flag = false;
                }
            }
        }
        // printf("number %d \n",number);
        
    }

    for (int i = 0; i < batch; ++i)
        free(idx[i]);
    free(idx);
    free(need_index);
}
FILE* file_U = fopen("dev_U.txt","w");
FILE* file_V = fopen("dev_V.txt","w");
FILE* file_diag = fopen("dev_diag.txt","w");
void test17(double* host_A,double* host_U,double* host_V,double* host_diag,int height,int width,int require_gpu,int total_batch,double& time_total,bool flag,int batch_size){
    int num_gpus;

    // 获取 GPU 数量
    cudaGetDeviceCount(&num_gpus);
    // printf("nums gpu is %d\n ",num_gpus);
    num_gpus=2;
    int gpu_group = num_gpus/require_gpu;
    
    int batch = 1;
    if(flag)
        batch = batch_size;
    printf("the batch is %d \n",batch);
    total_batch = batch * gpu_group;
    printf("the total batch %d \n",total_batch);
    int th=0, tw=0;
    // int shape[3] = {batch, height, width};
    int minmn = height > width/require_gpu ? width/require_gpu : height;

    // double* host_A = (double*)malloc(sizeof(double) * height * width*total_batch);
    // double* host_V = (double*)malloc(sizeof(double) * width*width*total_batch);
    string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

    // read in host A
    FILE* A_fp = fopen(matrix_path1.data(), "r");
    if(A_fp==NULL){
        generate_matrix(height, width);
        A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            // printf("open file falied\n");
            return ;
        }
    }
    for(int number=0;number < total_batch;++number){
        fseek(A_fp, 0, SEEK_SET);
        for(int i=0; i < height*width; i++){
            fscanf(A_fp, "%lf", &host_A[i+number*height*width]);
        }
    }

    fclose(A_fp);

    
    
    tw = 32;
    th = 32;
    int k = tw/2;
    int slice = th;
    int width_perdevice=width/require_gpu;
    size_t pitch;
    int*p_a,*p_b,*p_ab;

    // printf("input matrix shape: %d × %d × %d, tile shape: %d × %d\n", batch, height0, width0, th, tw);
    double t_start = omp_get_wtime();
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
    // double** dev_V0 = (double**)malloc(num_gpus * sizeof(double*));
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
    double** host_Fnorm = (double**)malloc(num_gpus * sizeof(double*));
    // int** host_order = (int**)malloc(num_gpus * sizeof(int*));
    // double** host_rawnorm = (double**)malloc(num_gpus*sizeof(double*));
    // double** host_norm = (double**)malloc(num_gpus*sizeof(double*));
    double** host_swap_data = (double**)malloc(sizeof(double*)*num_gpus);
    double** host_swap_V = (double**)malloc(sizeof(double*)*num_gpus);
    double** test_Fnorm = (double**)malloc(sizeof(double*)*num_gpus);
    // int gpuid = 0;
    for(int gpuid = 0;gpuid < num_gpus;++gpuid){
        p[gpuid]=(width_perdevice-1)/(2*k)+1;     
        host_swap_data[gpuid] = (double*)malloc(sizeof(double)*p[gpuid]*height*k*batch);
        host_swap_V[gpuid] = (double*)malloc(sizeof(double)*batch*width_perdevice*width/2);
        test_Fnorm[gpuid] = (double*)malloc(sizeof(double)*batch);
        host_allpass[gpuid] = (unsigned*)malloc(sizeof(unsigned)*batch);
    }
    int p2 = width_perdevice/k;
    q = (height-1)/slice+1;
    sliceNum = q;
#pragma endregion
cudaError_t err;
for(int gpuid = 0;gpuid < num_gpus;++gpuid){
    cudaSetDevice(gpuid);
        
    err = cudaMalloc((void**)&dev_pa[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pb[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab[gpuid],sizeof(int)*2*p[gpuid]*(2*p[gpuid]-1));
    // next_time
    cudaMalloc((void**)&dev_pa1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pb1[gpuid],sizeof(int)*p[gpuid]);
    cudaMalloc((void**)&dev_pab1[gpuid],sizeof(int)*2*p[gpuid]*p[gpuid]);

    cudaMalloc((void **)&dev_U[gpuid], sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_A[gpuid], sizeof(double) * height * width_perdevice * batch);
    err = cudaMalloc((void **)&dev_V[gpuid], sizeof(double) * width * width_perdevice * batch);
    // cudaMalloc((void **)&dev_V0[gpuid], sizeof(double) * width * width_perdevice * batch);
    cudaMalloc((void **)&dev_diag[gpuid],sizeof(double) * minmn*batch);
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
    err = cudaMemset(dev_V[gpuid], 0,  sizeof(double)*width * width_perdevice  * batch);
    cudaMemset(dev_U[gpuid], 0,  sizeof(double)*height * height * batch);
    cudaMemset(dev_diag[gpuid], 0,  sizeof(double)*minmn*batch);
    // cudaMemset(dev_V0[gpuid], 0, sizeof(double) * width * width_perdevice  * batch);
    cudaMemset(dev_pairsOfEVD[gpuid], 0,  sizeof(int) *2 * p[gpuid] * batch); 
    cudaMemset(dev_pass[gpuid], 0,  sizeof(unsigned)*p[gpuid] * batch);
}
    int shape[3]={batch,height,width_perdevice};


    unsigned int** host_order = (unsigned int**)malloc(sizeof(unsigned*)*gpu_group);
    double** host_rawnorm = (double**)malloc(sizeof(double*)*gpu_group);
    double** host_norm = (double**)malloc(sizeof(double*)*gpu_group);
    for(int i = 0;i < gpu_group;++i){
        host_order[i] = (unsigned int*)malloc(sizeof(unsigned int)*2*p[0]*require_gpu*batch);
        host_rawnorm[i] = (double*)malloc(sizeof(double)*p[0]*require_gpu*2*batch);
        host_norm[i] = (double*)malloc(sizeof(double)*p[0]*2*require_gpu*batch); 
    }

    cudaStream_t* stream = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    for(int j = 0;j < gpu_group;++j){
        for(int number=0;number < batch;++number){
            for(int i = 0;i<require_gpu;++i){
                cudaSetDevice(j*require_gpu+i);
                cudaMemcpyAsync(dev_A[j*require_gpu+i]+number*width_perdevice*height,host_A+(number+j*batch)*width*height+i*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyHostToDevice,stream[j*require_gpu+i]);
            }  
        }
        
    }
    // FILE* file11 = fopen("dev_V1.txt","w");
    //     FILE* file22 = fopen("dev_V2.txt","w");
    //     FILE* file33 = fopen("dev_V3.txt","w");
    //     FILE* file44 = fopen("dev_V4.txt","w");
    //     FILE* file_list[4]={file11,file22,file33,file44};
    //     for(int group_id = 0;group_id < gpu_group;++group_id){
    //         for(int number = 0;number < batch;++number){
    //             for(int gpuid = 0;gpuid < require_gpu;++gpuid){
    //                 cudaMemcpy(host_A+(group_id*batch+number)*height*width+gpuid*height*width_perdevice,dev_A[group_id*require_gpu+gpuid]+number*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost);
    //             }
    //         }   
    //     }
    //     for(int number=0;number<total_batch;++number){
    //         for(int w = 0;w < width;++w){
    //             for(int h=0;h<height;++h){
    //                 fprintf(file_list[0],"%f ",host_A[number*width*height+w*height+h]);
    //             }
    //             fprintf(file_list[0],"\n");
    //         }
    //     }
    //     return;
    // printf("the batch is %d \n",batch);
    // printf("width_per device %d \n",width_perdevice);
    omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        int Gpu_id = gpuid % require_gpu;
        Multi_init_dev_V<<<batch,256,0,stream[gpuid]>>>(dev_V[gpuid],width_perdevice,width,Gpu_id);
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

    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        getRankNewNew_2<<<1,1024,0,stream[gpuid]>>>(2*p[gpuid],dev_pab[gpuid],dev_pa[gpuid],dev_pb[gpuid]);
        getRankNewNew_1<<<1,1024,0,stream[gpuid]>>>(p[gpuid],dev_pa1[gpuid],dev_pb1[gpuid],dev_pab1[gpuid]);
    }
    int sweep = 0,maxsweep = 21;
    double svd_tol = 1e-9;
    if(num_gpus == 2){
        svd_tol = 1e-8;
    }
    int** raw_host_order = (int**)malloc(sizeof(int*)*gpu_group);

    for(int group_id=0;group_id<gpu_group;++group_id){
        raw_host_order[group_id] = (int*)malloc(sizeof(int)*2*p[0]*require_gpu*batch);
    }
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width_perdevice, p[gpuid], q, k);
        binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);  
    }
    for(int i = 0;i < gpu_group;++i){
        for(int gpuid = 0;gpuid < require_gpu;++gpuid)
        {
            cudaMemcpyAsync(host_order[i]+gpuid*2*p[i*require_gpu+gpuid]*batch,dev_order[i*require_gpu+gpuid],sizeof(unsigned int)*2*p[i*require_gpu+gpuid]*batch,cudaMemcpyDeviceToHost,stream[i*require_gpu+gpuid]);
            cudaMemcpyAsync(host_norm[i]+gpuid*2*p[i*require_gpu+gpuid]*batch,dev_norm[i*require_gpu+gpuid],sizeof(double)*2*p[i*require_gpu+gpuid]*batch,cudaMemcpyDeviceToHost,stream[i*require_gpu+gpuid]);
        }
       
    }
    
    for(int group_id = 0;group_id < gpu_group;++group_id){
        for(int gpuid = 0;gpuid < require_gpu;++gpuid){
            for(int number = 0;number < batch;++number){
                // printf("retirn\n");
                for(int p_index = 0;p_index < 2*p[0];++p_index){         
                    int offset_1 = gpuid*batch*2*p[0] + number * 2 * p[0];
                    // printf("offset %d \n",host_order[group_id][offset_1+p_index]);
                    host_rawnorm[group_id][offset_1+host_order[group_id][offset_1+p_index]] = host_norm[group_id][offset_1+p_index];
                }
            }
        }
    }
    
    int** host_index=(int**)malloc(sizeof(int*)*gpu_group);
    for(int group_id=0;group_id<gpu_group;++group_id){
        host_index[group_id] = (int*)malloc(sizeof(int)*2*p[0]*require_gpu*batch);
        fill_hostorder_total_new(raw_host_order[group_id],host_order[group_id],host_rawnorm[group_id],p,require_gpu,batch);

        for(int number=0;number<batch;++number){
            for(int i = 0;i < 2*p[0]*require_gpu;++i){
                host_index[group_id][raw_host_order[group_id][number*2*p[0]*require_gpu+i]+number*2*p[0]*require_gpu] = i;
            }
        }
    }
    // printf("host index \n");
    // for(int group_id=0;group_id<gpu_group;++group_id){
    //     for(int u = 0;u < 5;++u){
    //         printf("%d ",host_index[group_id][u]);
    //     }
    //     printf("\n");
    // }
    // return;
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
        computeFnorm2<<<batch, 32,0,stream[gpuid]>>>(dev_tempFnorm[gpuid], dev_Fnorm[gpuid], p[gpuid]);  //&1.3
    }
    // // printf("cudamemcpy error start \n");
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaMemcpy(test_Fnorm[i],dev_Fnorm[i],sizeof(double)*batch,cudaMemcpyDeviceToHost);
    }
    // // printf("cudaMemcpy error Fnorm \n");
    double** Fin_Fnorm = (double**)malloc(sizeof(double*)*gpu_group);
    for(int group_id=0;group_id<gpu_group;++group_id){
        Fin_Fnorm[group_id] = (double*)malloc(sizeof(double)*batch);
    }
    for(int group_id=0;group_id<gpu_group;++group_id){
        for(int bat = 0;bat < batch;++bat){
            for(int i = 0;i < require_gpu;++i)
                Fin_Fnorm[group_id][bat] += test_Fnorm[i+group_id*require_gpu][bat];
        }
    }

    for(int group_id=0;group_id<gpu_group;++group_id){
        for(int gpuid = 0;gpuid < require_gpu;++gpuid){
            cudaSetDevice(group_id*require_gpu+gpuid);
            cudaMemcpyAsync(dev_Fnorm[group_id*require_gpu+gpuid],Fin_Fnorm[group_id],sizeof(double)*batch,cudaMemcpyHostToDevice,stream[group_id*require_gpu+gpuid]);
        }
    }

    int** host_order_total = (int**)malloc(sizeof(int*)*gpu_group);
    for(int group_id=0;group_id<gpu_group;++group_id){
        host_order_total[group_id] = (int*)malloc(sizeof(int)* 2*p[0]*require_gpu*batch);
    }
    
        // for(int i = 0;i < num_gpus;++i){
        //     for(int j = 0;j < width_perdevice;++j){
        //         for(int r = 0;r < height;++r){
        //             fprintf(file_list[i],"%f ",host_A[i*height*width_perdevice+j*width+r]);
        //         }
        //         fprintf(file_list[i],"\n");
        //     }
        // }
        // return;
     
    while(!continue_flag){ 
        // part1
        dim3 dimGrid77(sliceNum, p[0], batch);// 2×2×100个block，每个block 256线程
        dim3 dimGrid7(p[0], batch, 1);
        clock_t rotate_1 = clock(); 
        omp_set_num_threads(num_gpus);
        for(int i = 0;i < 2*p[0]-1;++i){
            #pragma omp parallel
            {
                int gpuid = omp_get_thread_num();
                // int gpuid = 0;
                cudaSetDevice(gpuid);
                generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i, k, slice, sliceNum);    //&1.3

                generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height,width, width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3    
            }
        }
        
        
        
        clock_t rotate_2 = clock();
        printf("rotate time %f %d \n",(double)(rotate_2-rotate_1)/CLOCKS_PER_SEC,2*p[0]);
        int init_time = 1;
        while(init_time < require_gpu){
            for(int total_time = 1;total_time < require_gpu / init_time;++total_time){
                for(int group_id=0;group_id<gpu_group;++group_id){
                        for(int i = 0;i<require_gpu;++i){
                            cudaSetDevice(i+group_id*require_gpu);
                            for(int number=0;number < batch;++number){
                                cudaMemcpyAsync(host_swap_data[i+group_id*require_gpu]+number * width_perdevice*height/2,dev_A[i+group_id*require_gpu]+number * width_perdevice * height + width_perdevice*height/2,sizeof(double)*width_perdevice*height/2,cudaMemcpyDeviceToHost,stream[i+group_id*require_gpu]);             
                                cudaMemcpyAsync(host_swap_V[i+group_id*require_gpu] + number* width * width_perdevice/2,dev_V[i+group_id*require_gpu]+number * width*width_perdevice+width*width_perdevice/2,sizeof(double)*width*width_perdevice/2,cudaMemcpyDeviceToHost,stream[i+group_id*require_gpu]);
                            }  
                        }

                        for(int per_time = 1;per_time <= init_time;++per_time){
                            for(int i = require_gpu/init_time * (per_time-1);i < require_gpu/init_time * per_time;++i){
                                cudaSetDevice(i+group_id*require_gpu);
                                if(i == require_gpu/init_time * per_time-1){
                                    for(int number=0;number < batch;++number){
                                        cudaMemcpyAsync(dev_A[i+group_id*require_gpu]+number * height*width_perdevice+width_perdevice*height/2,host_swap_data[require_gpu/init_time*(per_time-1)+group_id*require_gpu]+number*width_perdevice*height/2,sizeof(double)*width_perdevice*height/2,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                                        cudaMemcpyAsync(dev_V[i+group_id*require_gpu]+number * width_perdevice*width+width*width_perdevice/2,host_swap_V[require_gpu/init_time*(per_time-1)+group_id*require_gpu]+number*width*width_perdevice/2,sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                                    }  
                                }   
                                else{
                                    for(int number=0;number < batch;++number){
                                        cudaMemcpyAsync(dev_A[i+group_id*require_gpu]+number * height*width_perdevice+width_perdevice*height/2,host_swap_data[i+1+group_id*require_gpu]+number*width_perdevice*height/2,sizeof(double)*width_perdevice*height/2,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                                        cudaMemcpyAsync(dev_V[i+group_id*require_gpu]+number * width_perdevice*width+width*width_perdevice/2,host_swap_V[i+1+group_id*require_gpu]+number*width*width_perdevice/2,sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                                    }
                                }    
                            }
                    }     
                }   
                 
                for(int i = 0;i < p[0];++i){
                    #pragma omp parallel
                    {
                        int gpuid = omp_get_thread_num();
                        cudaSetDevice(gpuid);
                        generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab1[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i, k, slice, sliceNum);    //&1.3
                        generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                        MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width,width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
                    }    
                }        
            }
            init_time *= 2;
            if(init_time <= require_gpu){
                for(int group_id=0;group_id<gpu_group;++group_id){
                    for(int i = 1;i <= init_time;++i){
                        int diff = i%2==1?width_perdevice*height/2:0;
                        int diff_V = i%2 == 1?width_perdevice*width/2:0;
                        for(int g = require_gpu/init_time*(i-1);g < require_gpu/init_time * i;++g){
                            cudaSetDevice(g+group_id*require_gpu);
                            for(int number = 0;number < batch;++number){
                                cudaMemcpyAsync(host_swap_data[g+group_id*require_gpu]+number*width_perdevice*height/2,dev_A[g+group_id*require_gpu]+number*width_perdevice*height+diff,sizeof(double)*width_perdevice*height/2,cudaMemcpyDeviceToHost,stream[g+group_id*require_gpu]);
                                cudaMemcpyAsync(host_swap_V[g+group_id*require_gpu]+number*width*width_perdevice/2,dev_V[g+group_id*require_gpu]+number*width_perdevice*width+diff_V,sizeof(double)*width_perdevice*width/2,cudaMemcpyDeviceToHost,stream[g+group_id*require_gpu]);
                            }    
                        }                             
                    }     
                    for(int i = 1;i <= init_time;++i){
                        int diff = i%2==1?width_perdevice*height/2:0;
                        int diff_V = i%2 == 1?width_perdevice*width/2:0;
                        int flag = i%2;
                        // printf("loop start \n");
                        for(int g = require_gpu/init_time*(i-1);g < require_gpu/init_time * i;++g){
                                cudaSetDevice(g+group_id*require_gpu);
                                if(flag == 0){
                                    for(int number=0;number < batch;++number){
                                        // printf("flag the need id %d \n",g+group_id*require_gpu);
                                        cudaMemcpyAsync(dev_A[g+group_id*require_gpu]+number * height*width_perdevice+diff,host_swap_data[g-require_gpu/init_time+group_id*require_gpu]+number*width_perdevice*height/2,sizeof(double)*width_perdevice*height/2,cudaMemcpyHostToDevice,stream[g+group_id*require_gpu]);
                                        cudaMemcpyAsync(dev_V[g+group_id*require_gpu]+number * width_perdevice*width+diff_V,host_swap_V[g-require_gpu/init_time+group_id*require_gpu]+number*width*width_perdevice/2,sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[g+group_id*require_gpu]);
                                    }
                                }
                                else{
                                    for(int number=0;number < batch;++number){
                                        // printf("the need id %d \n",g+group_id*require_gpu);
                                        cudaMemcpyAsync(dev_A[g+group_id*require_gpu]+number * height*width_perdevice+diff,host_swap_data[g+require_gpu/init_time+group_id*require_gpu]+number*width_perdevice*height/2,sizeof(double)*width_perdevice*height/2,cudaMemcpyHostToDevice,stream[g+group_id*require_gpu]);
                                        cudaMemcpyAsync(dev_V[g+group_id*require_gpu]+number * width_perdevice*width+diff_V,host_swap_V[g+require_gpu/init_time+group_id*require_gpu]+number*width*width_perdevice/2,sizeof(double)*width*width_perdevice/2,cudaMemcpyHostToDevice,stream[g+group_id*require_gpu]);
                                    }
                                }
                            }       
                        }            
                }    
                for(int i = 0;i < p[0];++i){
                    #pragma omp parallel
                    {
                        int gpuid = omp_get_thread_num();
                        cudaSetDevice(gpuid);
                        generate_jointG00_1<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_pab1[gpuid],dev_A[gpuid], height, width_perdevice, p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid],i,k, slice, sliceNum);    //&1.3
                        generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                        MUL_EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height, width,width_perdevice, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3
                    }    
                }  
            }
                       
        }
        
        
        // for(int number = 0;number < batch;++number){
        //     for(int i = 0;i < num_gpus;++i){
        //         cudaSetDevice(i);
        //         cudaMemcpyAsync(host_A+number*width*height+i*width_perdevice*height,dev_A[i]+number*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream[i]);
        //         cudaMemcpyAsync(host_V+number*width*width+i*width_perdevice*width,dev_V[i]+number*width_perdevice*width,sizeof(double)*width_perdevice*width,cudaMemcpyDeviceToHost,stream[i]);
        //     }
        // }
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            // printf("gpuid %d \n",gpuid);
            compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width_perdevice, p[gpuid], q, k);
            binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);
        } 
        for(int i = 0;i < gpu_group;++i){
            for(int j = 0;j < require_gpu;++j){
                cudaMemcpyAsync(host_order[i]+j*2*p[i]*batch,dev_order[i*require_gpu+j],sizeof(unsigned int)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
                cudaMemcpyAsync(host_norm[i]+j*2*p[i]*batch,dev_norm[i*require_gpu+j],sizeof(double)*2*p[i]*batch,cudaMemcpyDeviceToHost,stream[i]);
            }
        }
        


        for(int group_id = 0;group_id < gpu_group;++group_id){
            for(int gpuid = 0;gpuid < require_gpu;++gpuid){
                for(int number = 0;number < batch;++number){
                    for(int p_index = 0;p_index < 2*p[0];++p_index){
                        int offset_1 = gpuid*batch*2*p[0] + number * 2 * p[0];
                        host_rawnorm[group_id][offset_1+host_order[group_id][offset_1+p_index]] = host_norm[group_id][offset_1+p_index];
                    }
                }
            }
        }    
        for(int group_id=0;group_id<gpu_group;++group_id){
            fill_hostorder_total_new(host_order_total[group_id],host_order[group_id],host_rawnorm[group_id],p,require_gpu,batch);
        }

        
       
        
        for(int group_id = 0;group_id < gpu_group;++group_id){
            for(int number = 0;number < batch;++number){
                for(int gpuid = 0;gpuid < require_gpu;++gpuid){
                    cudaSetDevice(group_id*require_gpu+gpuid);
                    cudaMemcpyAsync(host_A+(number+group_id*batch)*width*height+gpuid*width_perdevice*height,dev_A[group_id*require_gpu+gpuid]+number*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream[group_id*require_gpu+gpuid]);
                    cudaMemcpyAsync(host_V+(number+group_id*batch)*width*width+gpuid*width_perdevice*width,dev_V[group_id*require_gpu+gpuid]+number*width_perdevice*width,sizeof(double)*width_perdevice*width,cudaMemcpyDeviceToHost,stream[group_id*require_gpu+gpuid]);
                }
            }
        }

    //    FILE* fileof = fopen("devoffser.txt","w");
        for(int group_id=0;group_id<gpu_group;++group_id){
            int cnt = 0;
            for(int number = 0;number < batch;++number){
                for(int i = 0;i<require_gpu;++i){
                    cudaSetDevice(i+group_id*require_gpu);
                    int per_len = number*2*p[0];
                    // printf("begin to detect\n");
                    for(int index = 0;index < 2*p[0];++index){ 
                        // printf("host order total\n  %d ",host_order_total[group_id][0]);
                        int offset = host_order_total[group_id][host_index[group_id][cnt]]+2*number*p[i]*require_gpu+group_id*2*p[i]*batch*require_gpu;
                        // printf("groupid: %d offset %d \n",group_id,offset);
                        // fprintf(fileof,"  groupid:%d offset:%d ",group_id,offset);
                        cudaMemcpyAsync(dev_A[i+group_id*require_gpu]+per_len*k*height,&host_A[offset*k*height],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                        cudaMemcpyAsync(dev_V[i+group_id*require_gpu]+per_len*k*width,&host_V[offset*k*width],sizeof(double)*k*width,cudaMemcpyHostToDevice,stream[i+group_id*require_gpu]);
                        ++cnt;
                        per_len++;
                    }
                    // fprintf(fileof,"\n ");
                    
                    
                    // printf("end detecting \n");
                }
            }

        }
        // FILE* fileu = fopen("devaaa.txt","w");
        // for(int group_id = 0;group_id < gpu_group;++group_id){
        //     for(int number = 0;number<batch;++number){
        //         for(int gpuid = 0;gpuid < require_gpu;++gpuid){
        //             cudaMemcpy(&host_A[group_id*batch*height*width+number*height*width+gpuid*width_perdevice*height],dev_A[group_id*require_gpu+gpuid]+number*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost);
        //         }
        //     }  
        // }
        // for(int i = 0;i < total_batch;++i){
        //     for(int g = 0;g < 10;++g){
        //         for(int u = 0;u < 10;++u){
        //             fprintf(fileu,"%f ",host_A[i*height*width+g*height+u]);
        //         }
        //         fprintf(fileu,"\n");
        //     } 
        //     for(int g = width_perdevice;g < width_perdevice+10;++g){
        //         for(int u = 0;u < 10;++u){
        //             fprintf(fileu,"%f ",host_A[i*height*width+g*height+u]);
        //         }
        //         fprintf(fileu,"\n");
        //     }  
        // }
        // return;
        
        
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
         int gpuid = omp_get_thread_num();
         cudaSetDevice(gpuid);
         dim3 dimGrid10(2 * p[gpuid], batch, 1);
         dim3 dimBlock10(32, k, 1);
         Mul_getUDV<<<dimGrid10, dimBlock10,0,stream[gpuid]>>>(dev_A[gpuid], dev_U[gpuid], dev_V[gpuid],  height, width_perdevice, height, width_perdevice,width, p[gpuid], height/32, dev_diag[gpuid], width_perdevice, k);  //&1.3
     }

    
    // for(int i = 0;i < total_batch;++i){
    //     printf("%f \n",host_V[i*width*width]);
    // }
     double t_end = omp_get_wtime();
    // //  printf("it costs %f s\n",(double)(c2-c1)/CLOCKS_PER_SEC); 
    //  printf("Total compute time = %.3f s\n", t_end - t_start);
     time_total += (t_end-t_start);
    // printf("width_perdevide %d  %d \n",width_perdevice,minmn);
     printf("sweep:%d \n",sweep);
    // for(int group_id=0;group_id<gpu_group;++group_id){
    //     for(int number=0;number < batch_size;++number){
    //         for(int gpuid=0;gpuid < require_gpu;++gpuid){
    //             cudaMemcpyAsync(&host_diag[group_id*batch_size*width+gpuid*width_perdevice+number*width],dev_diag[group_id*require_gpu+gpuid]+number*width_perdevice,sizeof(double)*width_perdevice,cudaMemcpyDeviceToHost,stream[group_id*require_gpu+gpuid]);
    //             cudaMemcpyAsync(&host_U[group_id*batch_size*width*height+gpuid*width_perdevice*height+number*width*height],dev_U[group_id*require_gpu+gpuid]+number*width_perdevice*height,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream[group_id*require_gpu+gpuid]);
    //             cudaMemcpyAsync(&host_V[group_id*batch_size*width*width+gpuid*width_perdevice*width+number*width*width],dev_V[group_id*require_gpu+gpuid]+number*width_perdevice*width,sizeof(double)*width_perdevice*width,cudaMemcpyDeviceToHost,stream[group_id*require_gpu+gpuid]);
    //         }
    //     }     
    // }
    //  double** host_diag = (double**)malloc(sizeof(double*)*num_gpus);
    //  for(int i = 0;i < num_gpus;++i){
    //      host_diag[i] = (double*)malloc(sizeof(double)*minmn*batch);
    //  }
    //  FILE* file2 = fopen("dev_diag.txt","w");
    //  for(int i = 0;i < num_gpus;++i){
    //         cudaMemcpy(host_diag[i],dev_diag[i],sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    //  }
    // for(int number = 0;number < batch;++number){
    //     for(int i = 0;i < num_gpus;++i){
    //         for(int g = 0;g < minmn;++g){
    //             fprintf(file2,"%lf ",host_diag[i][number*minmn+g]);
    //         }
    //     }
    //     if(number != batch-1)
    //     fprintf(file2,"\n");
    // }
    // FILE* file_V = fopen("dev_V.txt","w");
    // double** host_V11 = (double**)malloc(num_gpus * sizeof(double*));
    // for(int i = 0;i < num_gpus;++i){
    //     host_V11[i] = (double*)malloc(sizeof(double)*width_perdevice*width*batch);
    //     cudaMemcpy(host_V11[i],dev_V[i],sizeof(double)*width_perdevice*width*batch,cudaMemcpyDeviceToHost);
    // }
    // for(int group_id = 0;group_id<gpu_group;++group_id){
    //     for(int number = 0;number < batch;++number){
    //         for(int gpuid = 0;gpuid < require_gpu;++gpuid){
    //             for(int i  = 0;i < width_perdevice;++i){
    //                 for(int j =0;j < width;++j){
    //                     fprintf(file_V,"%lf ",host_V11[group_id*require_gpu+gpuid][number*width*width_perdevice+i*width+j]);
    //                 }
    //             fprintf(file_V,"\n");
    //         }
    //         }
    //     }
    // }
    // FILE* file_U = fopen("dev_U.txt","w");
    // double** host_U = (double**)malloc(num_gpus * sizeof(double*));
    // for(int i = 0;i < num_gpus;++i){
    //     host_U[i] = (double*)malloc(sizeof(double)*width_perdevice*height*batch);
    //     cudaMemcpy(host_U[i],dev_U[i],sizeof(double)*width_perdevice*height*batch,cudaMemcpyDeviceToHost);
    // }
    // for(int group_id = 0;group_id<gpu_group;++group_id){
    //     for(int number = 0;number < batch;++number){
    //         for(int gpuid = 0;gpuid < require_gpu;++gpuid){
    //             for(int i  = 0;i < width_perdevice;++i){
    //                 for(int j =0;j < height;++j){
    //                     fprintf(file_U,"%lf ",host_U[group_id*require_gpu+gpuid][number*height*width_perdevice+i*height+j]);
    //                 }
    //             fprintf(file_U,"\n");
    //         }
    //         }
    //     }
    // }
    // return;
    
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaFree(dev_A[i]);
        cudaFree(dev_U[i]);
        cudaFree(dev_V[i]);
        cudaFree(dev_pa[i]);
        cudaFree(dev_pb[i]);
        cudaFree(dev_pa1[i]);
        cudaFree(dev_pb1[i]);
        cudaFree(dev_pab[i]);
        cudaFree(dev_pab1[i]);
        cudaFree(dev_diag[i]);
        cudaFree(dev_roundRobin[i]);
        cudaFree(dev_jointG[i]);
        cudaFree(dev_Aij[i]);
        cudaFree(dev_AiAi[i]);
        cudaFree(dev_AiAj[i]);
        cudaFree(dev_AjAj[i]);
        cudaFree(dev_pairsOfEVD[i]);
        cudaFree(dev_swap_data[i]);
        cudaFree(dev_pass[i]);
        cudaFree(dev_norm[i]);
        cudaFree(dev_Fnorm[i]);
        cudaFree(dev_tempFnorm[i]);
        cudaFree(dev_order[i]);
        cudaFree(dev_allpass[i]);
        cudaStreamDestroy(stream[i]);
    }
    // free(host_A);
    // free(host_V);
    // free(host_U);
    // free(host_diag);
    for(int i = 0;i < gpu_group;++i){
        free(host_index[i]);
        free(raw_host_order[i]);
        free(host_order_total[i]);
        free(host_order[i]);
        free(Fin_Fnorm[i]);
    }
    for(int i = 0;i < num_gpus;++i){
        free(host_swap_data[i]);
        free(host_swap_V[i]);
        free(test_Fnorm[i]);
        free(host_allpass[i]);
    }
    free(host_index);
    free(raw_host_order);
    free(host_order_total);
    free(host_order);
    free(Fin_Fnorm);
    free(host_swap_data);
    free(host_swap_V);
    free(test_Fnorm);
    free(host_allpass);

}

void single_gpu_batch(double* host_A,double* host_U,double* host_V,double* host_diag,int width,int height,double& time_1,int batch_s){
   int num_gpus;

    // 获取 GPU 数量
    cudaGetDeviceCount(&num_gpus);
    // printf("nums gpu is %d\n ",num_gpus);
    // num_gpus=2;
    int batch = batch_s;
    int total_batch = batch * num_gpus;
    printf("the total batch %d \n",total_batch);
    int th=0, tw=0;
    // int shape[3] = {batch, height, width};
    int minmn = height > width ? width : height;

    // double* host_A = (double*)malloc(sizeof(double) * height * width*total_batch);
    // double* host_V = (double*)malloc(sizeof(double) * width*width*total_batch);
    string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

    // read in host A
    FILE* A_fp = fopen(matrix_path1.data(), "r");
    if(A_fp==NULL){
        generate_matrix(height, width);
        A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            // printf("open file falied\n");
            return ;
        }
    }
    for(int number=0;number < total_batch;++number){
        fseek(A_fp, 0, SEEK_SET);
        for(int i=0; i < height*width; i++){
            fscanf(A_fp, "%lf", &host_A[i+number*height*width]);
        }
    }

    fclose(A_fp);
    
    tw = 32;
    th = 32;
    int k = tw/2;
    int slice = th;
    size_t pitch;

    // printf("input matrix shape: %d × %d × %d, tile shape: %d × %d\n", batch, height0, width0, th, tw);
    double t_start = omp_get_wtime();
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
    double** dev_A = (double**)malloc(num_gpus * sizeof(double*));
    double** dev_V = (double**)malloc(num_gpus * sizeof(double*));
    // double** dev_V0 = (double**)malloc(num_gpus * sizeof(double*));
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
    double** host_Fnorm = (double**)malloc(num_gpus * sizeof(double*));
    // int** host_order = (int**)malloc(num_gpus * sizeof(int*));
    // double** host_rawnorm = (double**)malloc(num_gpus*sizeof(double*));
    // double** host_norm = (double**)malloc(num_gpus*sizeof(double*));
    double** test_Fnorm = (double**)malloc(sizeof(double*)*num_gpus);
    // int gpuid = 0;
    for(int gpuid = 0;gpuid < num_gpus;++gpuid){
        p[gpuid]=(width-1)/(2*k)+1;     
        test_Fnorm[gpuid] = (double*)malloc(sizeof(double)*batch);
        host_allpass[gpuid] = (unsigned*)malloc(sizeof(unsigned)*batch);
    }
    q = (height-1)/slice+1;
    sliceNum = q;
#pragma endregion
cudaError_t err;
for(int gpuid = 0;gpuid < num_gpus;++gpuid){
    cudaSetDevice(gpuid);
    cudaMalloc((void **)&dev_U[gpuid], sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_A[gpuid], sizeof(double) * height * width * batch);
    err = cudaMalloc((void **)&dev_V[gpuid], sizeof(double) * width * width * batch);
    // cudaMalloc((void **)&dev_V0[gpuid], sizeof(double) * width * width_perdevice * batch);
    cudaMalloc((void **)&dev_diag[gpuid],sizeof(double) * minmn*batch);
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
    err = cudaMemset(dev_V[gpuid], 0,  sizeof(double)*width * width  * batch);
    cudaMemset(dev_U[gpuid], 0,  sizeof(double)*height * height * batch);
    cudaMemset(dev_diag[gpuid], 0,  sizeof(double)*minmn*batch);
    // cudaMemset(dev_V0[gpuid], 0, sizeof(double) * width * width_perdevice  * batch);
    cudaMemset(dev_pairsOfEVD[gpuid], 0,  sizeof(int) *2 * p[gpuid] * batch); 
    cudaMemset(dev_pass[gpuid], 0,  sizeof(unsigned)*p[gpuid] * batch);
}
    int shape[3]={batch,height,width};
    unsigned int** host_order = (unsigned int**)malloc(sizeof(unsigned*)*num_gpus);
    double** host_norm = (double**)malloc(sizeof(double*)*num_gpus);
    for(int i = 0;i < num_gpus;++i){
        host_order[i] = (unsigned int*)malloc(sizeof(unsigned int)*2*p[0]*batch);
        host_norm[i] = (double*)malloc(sizeof(double)*p[0]*2*batch); 
    }
    cudaStream_t* stream = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
    }
    for(int gpuid = 0;gpuid < num_gpus;++gpuid){
        cudaMemcpyAsync(dev_A[gpuid],&host_A[gpuid*batch*height*width],sizeof(double)*batch*height*width,cudaMemcpyHostToDevice,stream[gpuid]);
    }

    omp_set_num_threads(num_gpus);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        init_dev_V<<<batch,256,0,stream[gpuid]>>>(dev_V[gpuid],width);
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

    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        getRankNewNew<<<1,1024,0,stream[gpuid]>>>(2*p[gpuid]);
    }
    int sweep = 0,maxsweep = 21;
    double svd_tol = 1e-7;
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width, p[gpuid], q, k);
        binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);  
    }

     #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(height >= 32){
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], height/32, height, width, k);
        }
        else{
            computeFnorm1<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_tempFnorm[gpuid], p[gpuid], 1, height, width, k);   
        }
        computeFnorm2<<<batch, 32,0,stream[gpuid]>>>(dev_tempFnorm[gpuid], dev_Fnorm[gpuid], p[gpuid]);  //&1.3
    }
    dim3 dimGrid77(sliceNum, p[0], batch);// 2×2×100个block，每个block 256线程
    dim3 dimGrid7(p[0], batch, 1);
    // FILE* file11 = fopen("dev_A1.txt","w");
    // FILE* file22 = fopen("dev_A2.txt","w");
    // FILE* file33 = fopen("dev_A3.txt","w");
    // FILE* file44 = fopen("dev_A4.txt","w");
    // FILE* file_list[4] = {file11,file22,file33,file44};
    
    while(!continue_flag){
        for(int i = 0;i < 2*p[0]-1;++i){
            #pragma omp parallel
            {
                int gpuid = omp_get_thread_num();
                // int gpuid = 0;
                cudaSetDevice(gpuid);
                generate_jointG00<<<dimGrid77, 256,0,stream[gpuid]>>>(dev_A[gpuid], height, width, dev_order[gpuid], dev_pass[gpuid], p[gpuid], q, dev_pairsOfEVD[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], i, k, slice, sliceNum);    //&1.3

                generate_jointG21<<<dimGrid7, 256,0,stream[gpuid]>>>(dev_jointG[gpuid], dev_AiAi[gpuid], dev_AiAj[gpuid], dev_AjAj[gpuid], dev_Fnorm[gpuid], dev_pass[gpuid], p[gpuid], k, sliceNum, svd_tol);    //&1.3
                EVD_1(stream[gpuid],dev_jointG[gpuid], dev_A[gpuid], dev_V[gpuid], dev_pairsOfEVD[gpuid], p[gpuid], q, height,width, dev_roundRobin[gpuid], batch, k, slice, sliceNum, sweep); //&1.3    
            }
        }
        // for(int i = 0;i <  num_gpus;++i){
        //     cudaMemcpy(&host_A[i*batch*height*width],dev_A[i],sizeof(double)*batch*height*width,cudaMemcpyDeviceToHost);
        // }
        // for(int i = 0;i <  num_gpus;++i){
        //     for(int number = 0;number<batch;++number){
        //         for(int w = 0;w < width;++w){
        //             for(int h = 0;h < height;++h){
        //                 fprintf(file_list[i],"%f ",host_A[i*batch*height*width+w*height+h+number*width*height]);
        //             }
        //             fprintf(file_list[i],"\n");
        //         }
        //     }    
        // }
        // return;
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            compute_norm<<<2 * p[gpuid] * batch, 128,0,stream[gpuid]>>>(dev_A[gpuid], dev_norm[gpuid], dev_order[gpuid], height, width, p[gpuid], q, k);
            binoticSort_original<<<batch, 1024,0,stream[gpuid]>>>(dev_norm[gpuid], dev_order[gpuid], 2 * p[gpuid], p[gpuid]);  
        }
        for(int gpuid=0;gpuid<num_gpus;++gpuid){
            cudaMemcpy(host_allpass[gpuid], dev_allpass[gpuid], sizeof(unsigned) * batch, cudaMemcpyDeviceToHost);
        }
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            judgeFunc<<<batch, 1024,0,stream[gpuid]>>>(dev_allpass[gpuid], dev_pass[gpuid], p[gpuid]);   // concentrate each block's result(converged or not)
        }
        bool tempFlag = true;
        for(int i = 0;i < num_gpus;++i){
            // printf("%d ",host_allpass[i][0]);
            if(!ifallpass(host_allpass[i],batch,p[i])){
                tempFlag = false;
            }
        }
        continue_flag = (tempFlag || sweep>maxsweep);  
        sweep++;
        printf("the sweep %d \n",sweep);
        for(int g = 0;g < num_gpus;++g){
            cudaSetDevice(g);
            cudaStreamSynchronize(stream[g]);
        } 
    }
    dim3 dimGrid10(2 * p[0], batch, 1);
    dim3 dimBlock10(32, k, 1);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        getUDV<<<dimGrid10, dimBlock10,0,stream[gpuid]>>>(dev_A[gpuid], dev_U[gpuid], dev_V[gpuid], height, width, height, width, p[gpuid], height/32, dev_diag[gpuid], width, k);  //&1.3
    }
    double t_end = omp_get_wtime();
    time_1 += (t_end-t_start);

    for(int gpuid = 0;gpuid < num_gpus;++gpuid){
        cudaMemcpyAsync(&host_diag[gpuid*batch_s*width],dev_diag[gpuid],sizeof(double)*width*batch_s,cudaMemcpyDeviceToHost,stream[gpuid]);
        cudaMemcpyAsync(&host_U[gpuid*batch_s*width*height],dev_U[gpuid],sizeof(double)*width*height*batch_s,cudaMemcpyDeviceToHost,stream[gpuid]);
        cudaMemcpyAsync(&host_V[gpuid*batch_s*width*width],dev_V[gpuid],sizeof(double)*width*width*batch_s,cudaMemcpyDeviceToHost,stream[gpuid]);
    }
    //  FILE* file2 = fopen("dev_diag.txt","w");
    //  for(int i = 0;i < num_gpus;++i){
    //         cudaMemcpy(host_diag[i],dev_diag[i],sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    //  }
    
    // FILE* file_V = fopen("dev_V.txt","w");
    // double** host_V11 = (double**)malloc(num_gpus * sizeof(double*));
    // for(int i = 0;i < num_gpus;++i){
    //     host_V11[i] = (double*)malloc(sizeof(double)*width_perdevice*width*batch);
    //     cudaMemcpy(host_V11[i],dev_V[i],sizeof(double)*width_perdevice*width*batch,cudaMemcpyDeviceToHost);
    // }
    // for(int group_id = 0;group_id<gpu_group;++group_id){
    //     for(int number = 0;number < batch;++number){
    //         for(int gpuid = 0;gpuid < require_gpu;++gpuid){
    //             for(int i  = 0;i < width_perdevice;++i){
    //                 for(int j =0;j < width;++j){
    //                     fprintf(file_V,"%lf ",host_V11[group_id*require_gpu+gpuid][number*width*width_perdevice+i*width+j]);
    //                 }
    //             fprintf(file_V,"\n");
    //         }
    //         }
    //     }
    // }
    // FILE* file_U = fopen("dev_U.txt","w");
    // double** host_U = (double**)malloc(num_gpus * sizeof(double*));
    // for(int i = 0;i < num_gpus;++i){
    //     host_U[i] = (double*)malloc(sizeof(double)*width_perdevice*height*batch);
    //     cudaMemcpy(host_U[i],dev_U[i],sizeof(double)*width_perdevice*height*batch,cudaMemcpyDeviceToHost);
    // }
    // for(int group_id = 0;group_id<gpu_group;++group_id){
    //     for(int number = 0;number < batch;++number){
    //         for(int gpuid = 0;gpuid < require_gpu;++gpuid){
    //             for(int i  = 0;i < width_perdevice;++i){
    //                 for(int j =0;j < height;++j){
    //                     fprintf(file_U,"%lf ",host_U[group_id*require_gpu+gpuid][number*height*width_perdevice+i*height+j]);
    //                 }
    //             fprintf(file_U,"\n");
    //         }
    //         }
    //     }
    // }
    for(int i = 0;i < num_gpus;++i){
        cudaSetDevice(i);
        cudaFree(dev_A[i]);
        cudaFree(dev_U[i]);
        cudaFree(dev_V[i]);
        cudaFree(dev_diag[i]);
        cudaFree(dev_roundRobin[i]);
        cudaFree(dev_jointG[i]);
        cudaFree(dev_Aij[i]);
        cudaFree(dev_AiAi[i]);
        cudaFree(dev_AiAj[i]);
        cudaFree(dev_AjAj[i]);
        cudaFree(dev_pairsOfEVD[i]);
        cudaFree(dev_pass[i]);
        cudaFree(dev_norm[i]);
        cudaFree(dev_Fnorm[i]);
        cudaFree(dev_tempFnorm[i]);
        cudaFree(dev_order[i]);
        cudaFree(dev_allpass[i]);
        cudaStreamDestroy(stream[i]);
    }
    // free(host_A);
    // free(host_V);
    // free(host_U);
    // free(host_diag);
    free(test_Fnorm);
    free(host_allpass);
    free(host_order);
    
}

double calculate(int width,int height,int require_gpu){
    double result_MB = 0;
    long int UAV_sum = height*height + height * width+width*width*require_gpu+width;
    long int other_sum = 32*width+width*height+height*width*3/4+width/32*2;
    long int int_sum = width/32+width/16*(width/16-1)+width/32+width/32+width/16*width/32+32*31+width/16+width/32+width/16;

    result_MB += 1.0 * sizeof(double)*UAV_sum/(1024*1024)+sizeof(double)*other_sum/(1024*1024)+int_sum*sizeof(int)*1.0/(1024*1024);
    return result_MB;
}
int main(int argc, char* argv[]){
    int num_gpus;
    // 获取 GPU 数量
    cudaGetDeviceCount(&num_gpus);
    num_gpus = 2;
    int height = 18432;
    int width = 18432;
    
    cudaDeviceProp prop;
    // 获取设备属性
    cudaError_t status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(status));
        return 1;
    }
    
    // 打印全局内存大小（单位：字节）
    printf("GPU Global Memory Size: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    double threshold = prop.totalGlobalMem / (1024 * 1024);
    int batch = 4;
    int require_gpu = 2;
    double bytes = calculate(width/require_gpu,height,require_gpu);
    double my_requiremem = bytes;
    int width_tmp = width;
    double* host_A = (double*)malloc(sizeof(double)*height*width*batch);
    double* host_U = (double*)malloc(sizeof(double)*height*width*batch);
    double* host_V = (double*)malloc(sizeof(double)*width*width*batch);
    double* host_diag = (double*)malloc(sizeof(double)*min(height,width)*batch);
    while(my_requiremem > threshold*0.9){
        width_tmp /= 2;
        require_gpu*=2;
        my_requiremem = calculate(height,width_tmp,require_gpu);
    }
    printf("require mem %f \n",my_requiremem);
    int batch_s = threshold*0.9/my_requiremem;
    printf("req %d \n",require_gpu);
    printf("req %d \n",batch_s);
    batch_s = min(batch_s,batch/(num_gpus/require_gpu));
    printf("req %d \n",batch_s);
    int circle_time = (batch-1)/(batch_s*num_gpus/require_gpu)+1;
    double time_1=0;
    printf("require gpu %d \n",require_gpu);
    printf("the circle timw %d \n",circle_time);
    if(require_gpu == 1){
        for(int i = 0;i < circle_time;++i){
            if(i*(num_gpus/require_gpu) + num_gpus/require_gpu <= batch)
                single_gpu_batch(host_A+i*batch_s*num_gpus/require_gpu*height*width,host_U+i*batch_s*num_gpus/require_gpu*height*width,host_V+i*batch_s*num_gpus/require_gpu*width*width,host_diag+i*batch_s*num_gpus/require_gpu*width,width,height,time_1,batch_s);
            else
                single_gpu_batch(host_A+i*batch_s*num_gpus/require_gpu*height*width,host_U+i*batch_s*num_gpus/require_gpu*height*width,host_V+i*batch_s*num_gpus/require_gpu*width*width,host_diag+i*batch_s*num_gpus/require_gpu*width,width,height,time_1,(batch-i*(num_gpus/require_gpu*batch_s))/num_gpus);
        }
    }
    else{
        bool is_test = true;
         for(int i = 0;i < circle_time;++i){
            if(i*(num_gpus/require_gpu) + num_gpus/require_gpu <= batch)
                test17(host_A+i*batch_s*num_gpus/require_gpu*height*width,host_U+i*batch_s*num_gpus/require_gpu*height*width,host_V+i*batch_s*num_gpus/require_gpu*width*width,host_diag+i*batch_s*num_gpus/require_gpu*width,width,height,require_gpu,num_gpus/require_gpu,time_1,is_test,batch_s);
            else
                test17(host_A+i*batch_s*num_gpus/require_gpu*height*width,host_U+i*batch_s*num_gpus/require_gpu*height*width,host_V+i*batch_s*num_gpus/require_gpu*width*width,host_diag+i*batch_s*num_gpus/require_gpu*width,width,height,require_gpu,batch-i*(num_gpus/require_gpu)*batch_s,time_1,is_test,(batch-i*(num_gpus/require_gpu)*batch_s)/(num_gpus/require_gpu));
            // break;
        }
    }

    // int require_gpu = 1;
    // int circle_time = batch / (num_gpus/require_gpu);
    
    // // double 
    // double time_1 = 0;
    // int width_tmp = width/2;
    // bool is_test = true;
    // int batch_s = 1;
    
    // if(!is_test){
    //     // printf("require memory: %f  \n",my_requiremem);
    //     while(my_requiremem > threshold*0.9)
    //     {
    //         width_tmp /= 2;
    //         require_gpu *= 2;
    //         my_requiremem = calculate(width_tmp,height);
    //     }
    //     require_gpu = 2;
    // }
    // else{
    //     // printf("require memory: %f  \n",my_requiremem);
    //     while(my_requiremem > threshold*0.9)
    //     {
    //         width_tmp /= 2;
    //         require_gpu *= 2;
    //         my_requiremem = calculate(width_tmp,height);
    //     }
    //     require_gpu = 2;
    //     width_tmp = width / require_gpu;
    //     circle_time = batch / (num_gpus/require_gpu);
    //     my_requiremem = calculate(width_tmp,height);
    //     batch_s = threshold*0.9/my_requiremem;
    //     // batch_s = 1;
    // }
    // // printf("require gpu number  %d \n",require_gpu);
    // // printf("circle time %d \n",circle_time);
    
    // batch_s = min(batch_s,batch/(num_gpus/require_gpu));
    // printf("circle_time %d \n",circle_time);
    // // return;
    // circle_time = circle_time / batch_s;
    // double t_start = omp_get_wtime();
    // // single_gpu_batch(1024,512,1);
    // printf("batch_s %d \n",batch_s);
    // printf("circle_time %d \n",circle_time);
   
    // double t_end = omp_get_wtime();
    printf("batch svd time %f \n",time_1);
    
    for(int number=0;number < batch;++number){
        for(int w = 0;w < width;++w){
            fprintf(file_diag,"%f ",host_diag[number*width+w]);
        }
        fprintf(file_diag,"\n");
        for(int w = 0;w < width;++w){
            for(int h =0;h < height;++h){
                fprintf(file_U,"%f ",host_U[number*width*height+w*height+h]);
            }
            fprintf(file_U,"\n");
        }
        
        for(int w = 0;w < width;++w){
            for(int h = 0;h < width;++h){
                fprintf(file_V,"%f ",host_V[number*width*width+w*width+h]);
            }
            fprintf(file_V,"\n");
        }
        

    }

    free(host_A);
    free(host_V);
    free(host_diag);
    free(host_U);

}