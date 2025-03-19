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



void fill_hostorder_total(int* host_order_total,int* host_order_1,int* host_order_2,double* host_norm_1,double* host_norm_2,int p,int p1){
    int index=0,idx1=0,idx2=0;
    while(idx1 < 2*p || idx2 < 2*p1){
        if(idx1 < 2*p && idx2 < 2*p1){
            if(host_norm_1[host_order_1[idx1]] < host_norm_2[host_order_2[idx2]]){
                host_order_total[index]=host_order_1[idx1];
                idx1++;
                index++;
            }
            else{
                host_order_total[index] = host_order_2[idx2]+2*p;
                idx2++;
                index++;
            }
        }
        else if(idx1 < 2*p){
            host_order_total[index] = host_order_1[idx1];
            idx1++;
            index++;
        }
        else{
            host_order_total[index] = host_order_2[idx2]+2*p;
            idx2++;
            index++;
        }
    }
}
// fig 14 a
// 100x512x512 speedup over cusolver(CUDA platform)
void test17(){
    int gpu0=0,gpu1=1;
    int batch = 1;
    int height = 2048;
    int width = 2048;
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

    tw = 32;
    th = 32;
    int k = tw/2;
    int slice = th;
    int width_perdevice=width/2;
    size_t pitch;
    int*p_a,*p_b,*p_ab;
   

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
    double* dev_V1;
    double* dev_U_1;
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
    cudaSetDevice(gpu0);
    cudaMalloc((void**)&p_a,sizeof(int)*(p+p1));
    cudaMalloc((void**)&p_b,sizeof(int)*(p+p1));
    cudaMalloc((void**)&p_ab,sizeof(int)*2*(p+p1)*(2*(p+p1)-1));

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
    cudaMalloc((void **)&dev_swap_data,sizeof(double)*p*height*k);

    host_allpass = (unsigned *)malloc(sizeof(unsigned) * batch);
    host_pass = (unsigned *)malloc(sizeof(unsigned) * p * batch);
    cudaMalloc((void **)&dev_allpass, sizeof(unsigned) * batch);
    cudaMalloc((void **)&dev_pass, sizeof(unsigned) * p * batch);

    cudaMalloc((void **)&dev_norm, sizeof(double) * 2 * p * batch);
    cudaMalloc((void **)&dev_order, sizeof(unsigned int) * 2 * p * batch);
    host_Fnorm = (double *)malloc(sizeof(double) * batch);
    cudaMalloc((void **)&dev_tempFnorm, sizeof(double) * 2 * p * batch);
    cudaMalloc((void **)&dev_Fnorm, sizeof(double) * batch);

#pragma endregion
#pragma region
    cudaSetDevice(gpu1);
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
    cudaMalloc((void **)&dev_swap_data_1,sizeof(double)*p1*height*k);

    host_allpass_1 = (unsigned *)malloc(sizeof(unsigned) * batch);
    host_pass_1 = (unsigned *)malloc(sizeof(unsigned) * p1 * batch);
    cudaMalloc((void **)&dev_allpass_1, sizeof(unsigned) * batch);
    cudaMalloc((void **)&dev_pass_1, sizeof(unsigned) * p1 * batch);

    cudaMalloc((void **)&dev_norm_1, sizeof(double) * 2 * p1 * batch);
    cudaMalloc((void **)&dev_order_1, sizeof(unsigned int) * 2 * p1 * batch);
    host_Fnorm_1 = (double *)malloc(sizeof(double) * batch);
    cudaMalloc((void **)&dev_tempFnorm_1, sizeof(double) * 2 * p1 * batch);
    cudaMalloc((void **)&dev_Fnorm_1, sizeof(double) * batch);

#pragma endregion

// preset before svd  
#pragma region
    cudaSetDevice(gpu0);
    cudaMemset(dev_V, 0, sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
    // cudaMemset(dev_V0, 0, sizeof(double) * width_perdevice * width_perdevice * batch);
    cudaMemset(dev_pairsOfEVD, 0, sizeof(int) * 2 * p * batch); 
    memset(host_pass, 0, sizeof(unsigned) * p * batch);
    cudaMemset(dev_pass, 0, sizeof(unsigned) * p * batch);
    int shape[3]={batch,height,width_perdevice};
    double* host_A1,* host_A2;
    host_A1 = (double*)malloc(sizeof(double)*width_perdevice*height);
    host_A2 = (double*)malloc(sizeof(double)*width_perdevice*height);
    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 2.0;
    cudaSetDevice(gpu0);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaSetDevice(gpu1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    cudaSetDevice(gpu0);
    cudaMemcpyAsync(dev_A,host_A,sizeof(double)*height*width_perdevice,cudaMemcpyHostToDevice,stream1);
    cudaSetDevice(gpu1);
    cudaMemcpyAsync(dev_A_1,host_A+height*width_perdevice,sizeof(double)*height*width_perdevice,cudaMemcpyHostToDevice,stream2);
    // cudaMemcpy(host_A2,dev_A_1,sizeof(double)*p1*k*height,cudaMemcpyDeviceToHost);
    double* test_A_1 = (double*)malloc(sizeof(double)*height*width_perdevice);
    cudaMemcpy(test_A_1,dev_A_1,sizeof(double)*height*width_perdevice,cudaMemcpyDeviceToHost);
    // printf("devA\n");
    // for(int g = 0;g < 5;++g){
    //     printf("%lf ",test_A_1[g]);
    // }

    cudaSetDevice(gpu0);
    dim3 dimGrid0(1, 1, 1);
    dim3 dimBlock0(32, 32, 1);
    generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream1>>>(dev_roundRobin, 2*k);
    cudaSetDevice(gpu1);
    generate_roundRobin_128<<<dimGrid0, dimBlock0,0,stream2>>>(dev_roundRobin_1, 2*k);
    bool continue_flag = false;
    int *host_order_1 = (int*)malloc(sizeof(int)*2*p*batch);
    int *host_order_2 = (int*)malloc(sizeof(int)*2*p1*batch);
    int *host_index_1 = (int*)malloc(sizeof(int)*2*p*batch);
    int *host_index_2 = (int*)malloc(sizeof(int)*2*p1*batch);
    int *host_order_total = (int*)malloc(sizeof(int)*2*(p+p1)*batch);
    double* host_norm_1 = (double*)malloc(sizeof(double)*2*p*batch);
    double* host_norm_2 = (double*)malloc(sizeof(double)*2*p1*batch);

    double* host_rawnorm_1 = (double*)malloc(sizeof(double)*2*p*batch);
    double* host_rawnorm_2 = (double*)malloc(sizeof(double)*2*p1*batch);

    omp_set_num_threads(2);
    // calculate the itertime's index pair
    cudaSetDevice(gpu0);
    getRankNewNew_2<<<1, 1024,0,stream1>>>(2 * (p+p1),p_ab,p_a,p_b);  //&1.3
    // getRankNewNew<<<1, 1024,0,stream2>>>(2 * p1);  //&1.3
    int* host_ab = (int*)malloc(sizeof(int)*2*(p+p1)*(2*(p+p1)-1));
    cudaMemcpyAsync(host_ab,p_ab,sizeof(int)*2*(p+p1)*(2*(p+p1)-1),cudaMemcpyDeviceToHost,stream1);
    // for(int i = 0;i < 2*(p+p1)-1;++i){
    //     for(int g = 0;g < 2*(p+p1);++g){
    //         printf("%d ",host_ab[i*(2*(p+p1))+g]);
    //     }
    //     printf("\n");
    // }
    // unsigned* host_allpass = (unsigned*)malloc(sizeof(batch));
    // unsigned* host_allpass_1 = (unsigned*)malloc(sizeof(batch));
    int sweep = 0,maxsweep = 11;
    double svd_tol = 1e-7;
    clock_t begin = clock();
    while(!continue_flag){
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            
            // printf("gpuid %d \n",gpuid);
            if(gpuid==0){
                cudaSetDevice(gpu0);
                compute_norm<<<2 * p * batch, 128,0,stream1>>>(dev_A, dev_norm, dev_order, height, width_perdevice, p, q, k);
                binoticSort_original<<<batch, 1024,0,stream1>>>(dev_norm, dev_order, 2 * p, p);
            }
            else{
                cudaSetDevice(gpu1);
                compute_norm<<<2 * p1 * batch, 128,0,stream2>>>(dev_A_1, dev_norm_1, dev_order_1, height, width_perdevice, p1, q, k);
                binoticSort_original<<<batch, 1024,0,stream2>>>(dev_norm_1, dev_order_1, 2 * p1, p1);
            }
        } 
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(host_order_1,dev_order,sizeof(int)*2*p*batch,cudaMemcpyDeviceToHost,stream1);
        cudaMemcpyAsync(host_norm_1,dev_norm,sizeof(double)*2*p*batch,cudaMemcpyDeviceToHost,stream1);
        cudaStreamSynchronize(stream2);
        cudaMemcpyAsync(host_order_2,dev_order_1,sizeof(int)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
        cudaMemcpyAsync(host_norm_2,dev_norm_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        #pragma omp parallel for
        for(int i = 0;i < 2*p;++i){
            // printf("%d ",host_order_1[i]);
            host_rawnorm_1[host_order_1[i]] = host_norm_1[i];
        }
        #pragma omp parallel for
        for(int i = 0;i < 2*p1;++i){
            // printf("%d ",host_order_2[i]);
            host_rawnorm_2[host_order_2[i]] = host_norm_2[i];
        }
        fill_hostorder_total(host_order_total,host_order_1,host_order_2,host_rawnorm_1,host_rawnorm_2,p,p1);
        // for(int f = 0;f < 2*(p+p1);++f){
        //     printf("%d ",host_order_total[f]);
        // }
        // printf("\n");
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            if(gpuid==0){
                if(height >= 32){
                    computeFnorm1<<<2 * p * batch, 128,0,stream1>>>(dev_A, dev_tempFnorm, p, height/32, height, width_perdevice, k);
                }
                else{
                    computeFnorm1<<<2 * p * batch, 128,0,stream1>>>(dev_A, dev_tempFnorm, p1, 1, height, width_perdevice, k);   
                }
                cudaDeviceSynchronize();
                computeFnorm2<<<batch, 32,0,stream1>>>(dev_tempFnorm, dev_Fnorm, p);  //&1.3
            }
            else{
                if(height >= 32){
                    computeFnorm1<<<2 * p1 * batch, 128,0,stream2>>>(dev_A_1, dev_tempFnorm_1, p1, height/32, height, width_perdevice, k);
                }
                else{
                    computeFnorm1<<<2 * p1 * batch, 128,0,stream2>>>(dev_A_1, dev_tempFnorm_1, p1, 1, height, width_perdevice, k);   
                }
                cudaDeviceSynchronize();
                computeFnorm2<<<batch, 32,0,stream2>>>(dev_tempFnorm_1, dev_Fnorm_1, p);  //&1.3
            }
        }
       

        double* test_Fnorm = (double*)malloc(sizeof(double)*batch);
        cudaMemcpyAsync(test_Fnorm,dev_Fnorm,sizeof(double)*batch,cudaMemcpyDeviceToHost,stream1);
        // printf("\nstream 1 test_Fnorm:   %lf\n",test_Fnrom[0]);
  
        double* test_Fnorm_1 = (double*)malloc(sizeof(double)*batch);
        cudaMemcpyAsync(test_Fnorm_1,dev_Fnorm_1,sizeof(double)*batch,cudaMemcpyDeviceToHost,stream2);
        // printf("\nstream 2 test_Fnorm:   %lf\n",test_Fnrom_1[0]);
        // printf("order index \n");
        double* Fin_Fnorm = (double*)malloc(sizeof(double)*batch);
        for(int bat = 0;bat < batch;++bat){
            Fin_Fnorm[bat] = test_Fnorm[bat]+test_Fnorm_1[bat];
        }
        cudaMemcpyAsync(dev_Fnorm,Fin_Fnorm,sizeof(double)*batch,cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(dev_Fnorm_1,Fin_Fnorm,sizeof(double)*batch,cudaMemcpyHostToDevice,stream2);
        
        // for(int u = 0;u < 2*(p+p1);++u){
        //     if(host_order_total[u]<2*p){
        //         printf("%lf ",host_rawnorm_1[host_order_total[u]]);
        //     }
        //     else{
        //         printf("%lf ",host_rawnorm_2[host_order_total[u]-2*p]);
        //     }
        // }
        // printf("\n");
        for(int iter = 0;iter < 2*p+2*p1-1;++iter){
            // memory distribute strategy
            // printf("raw host_A\n");
            // for(int g = 0;g < 4;++g){
            //     for(int t = 0;t < 5;++t){
            //         printf("%lf ",host_A[g*k*height+t]);
            //     }
            //     printf("\n");
            // }
            // printf("order1 \n");
            #pragma omp parallel for
            for(int i = 0;i < 2*p;++i){
                // printf("%d ",host_order_total[host_ab[iter*(2*(p+p1))+i]]);
                cudaMemcpyAsync(dev_A+i*k*height,host_A+k*height*host_order_total[host_ab[iter*(2*(p+p1))+i]],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream1);
                // cudaMemcpyAsync(dev_A_1+,dev_order_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
            }
            // printf("\norder2 \n");
            #pragma omp parallel for
            for(int i = 0;i < 2*p1;++i){
                // printf("%d ",host_order_total[host_ab[iter*(2*(p+p1))+i+2*p]]);
                cudaMemcpyAsync(dev_A_1+i*k*height,host_A+k*height*host_order_total[host_ab[iter*(2*(p+p1))+i+2*p]],sizeof(double)*k*height,cudaMemcpyHostToDevice,stream2);
                // cudaMemcpyAsync(dev_A_1+,dev_order_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
            }
            // printf("\n");

            double* test_host_A = (double*)malloc(sizeof(double)*width_perdevice*height);
            cudaMemcpyAsync(test_host_A,dev_A,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost,stream1);
            cudaStreamSynchronize(stream1);
            #pragma region
            // printf("dev mat\n");
            // for(int g = 0;g < 2;++g){
            //     for(int t = 0;t < 5;++t){
            //         printf("%lf ",test_host_A[g*k*height+t]);
            //     }
            //     printf("\n");
            // }
            // break;
            // printf("host_A\n");
            // for(int i = 0;i < 2;++i){
            //     for(int t = 0;t < 5;++t){
            //         printf("%lf ",host_A[k*height*host_order_total[host_ab[]]])
            //     }
            // }

            // debug1 dev_A
            // double* test_host_A = (double*)malloc(sizeof(double)*width_perdevice*height);
            // double* test_host_A_1 = (double*)malloc(sizeof(double)*width_perdevice*height);
            // cudaMemcpy(test_host_A,dev_A,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost);
            // cudaMemcpy(test_host_A_1,dev_A_1,sizeof(double)*width_perdevice*height,cudaMemcpyDeviceToHost);
            // for(int  r = 0;r < 5;++r){
            //     printf("%lf %lf ",test_host_A[r],test_host_A_1[r]);
            // }
            // printf("\n");
            // for(int g = 0;g < 2*(p+p1);++g){
            //     printf("%d ",host_ab[iter*(2*(p+p1))+g]);
            // }
            // printf("\n");

            // compute evd
            // every pair is back to back
            #pragma endregion
            #pragma omp parallel
            {
                int gpuid = omp_get_thread_num();
                cudaSetDevice(gpuid);
                dim3 dimGrid77(sliceNum, p, batch);// 2×2×100个block，每个block 256线程
                dim3 dimGrid7(p, batch, 1);
                if(gpuid==0){
                    generate_jointG00_2<<<dimGrid77, 256,0,stream1>>>(dev_A, height, width_perdevice, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj,  k, slice, sliceNum);    //&1.3
                    generate_jointG21<<<dimGrid7, 256,0,stream1>>>(dev_jointG, dev_AiAi, dev_AiAj, dev_AjAj, dev_Fnorm, dev_pass, p, k, sliceNum, svd_tol);    //&1.3
                    EVD_1(stream1,dev_jointG, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width_perdevice, dev_roundRobin, batch, k, slice, sliceNum, sweep); //&1.3
                }
                else{
                    generate_jointG00_2<<<dimGrid77, 256,0,stream2>>>(dev_A_1, height, width_perdevice, p1, q, dev_pairsOfEVD_1, dev_AiAi_1, dev_AiAj_1, dev_AjAj_1,  k, slice, sliceNum);    //&1.3
                    generate_jointG21<<<dimGrid7, 256,0,stream2>>>(dev_jointG_1, dev_AiAi_1, dev_AiAj_1, dev_AjAj_1, dev_Fnorm_1, dev_pass_1, p1, k, sliceNum, svd_tol);    //&1.3
                    EVD_1(stream2,dev_jointG_1, dev_A_1, dev_V_1, dev_pairsOfEVD_1, p1, q, height, width_perdevice, dev_roundRobin_1, batch, k, slice, sliceNum, sweep); //&1.3
                }
            }
#pragma region         
            

            // debug dev_AiAi
            // double* test_AiAi = (double*)malloc(sizeof(double)* k * k * sliceNum * p * batch);
            // cudaMemcpy(test_AiAi,dev_AiAi,sizeof(double)*k * k * sliceNum * p * batch,cudaMemcpyDeviceToHost);
            // printf("\ntest_AiAI\n");
            // for(int g = 0;g < 5;++g){
            //     printf("%lf ",test_AiAi[g]);
            // }
            // printf("\n");
           
           
            // cudaDeviceSynchronize();
            // debuging dev_jointG
            // double* test_jointG = (double*)malloc(sizeof(double)*2*k*2*k*p*batch);
            // cudaMemcpy(test_jointG,dev_jointG,sizeof(double)*2*k*2*k*p*batch,cudaMemcpyDeviceToHost);
            // printf("\n test_jointG\n");
            // for(int f = 0;f < 5;++f){
            //     printf("%lf ",test_jointG[f]);
            // }
            // printf("\n");
            
#pragma endregion            
            cudaDeviceSynchronize();
            #pragma omp parallel
            for(int i = 0;i < 2*p;++i){
                // printf("%d ",host_order_total[host_ab[iter*(2*(p+p1))+i]]);
                cudaMemcpyAsync(host_A+k*height*host_order_total[host_ab[iter*(2*(p+p1))+i]],dev_A+i*k*height,sizeof(double)*k*height,cudaMemcpyDeviceToHost,stream1);
                // cudaMemcpyAsync(dev_A_1+,dev_order_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
            }
            // printf("\n");
            #pragma omp parallel
            for(int i = 0;i < 2*p1;++i){
                // printf("%d ",host_order_total[host_ab[iter*(2*(p+p1))+i+2*p]]);
                cudaMemcpyAsync(host_A+k*height*host_order_total[host_ab[iter*(2*(p+p1))+i+2*p]],dev_A_1+i*k*height,sizeof(double)*k*height,cudaMemcpyDeviceToHost,stream2);
                // cudaMemcpyAsync(dev_A_1+,dev_order_1,sizeof(double)*2*p1*batch,cudaMemcpyDeviceToHost,stream2);
            }

            #pragma region
            // printf("pass\n");

            // printf("\n stream1 host_pass  %d \n",p1);
            unsigned* test_pass = (unsigned*)malloc(sizeof(unsigned)*p1*batch);
            cudaMemcpy(test_pass,dev_pass,sizeof(unsigned)*p1*batch,cudaMemcpyDeviceToHost);
            // for(int f = 0;f < p1*batch;++f){
            //     printf("%d ",test_pass[f]);
            // }
            // printf("\n stream2 host_pass  %d \n",p1);
            unsigned* test_pass_1 = (unsigned*)malloc(sizeof(unsigned)*p1*batch);
            cudaMemcpy(test_pass_1,dev_pass_1,sizeof(unsigned)*p1*batch,cudaMemcpyDeviceToHost);
            // for(int f = 0;f < p1*batch;++f){
            //     printf("%d ",test_pass_1[f]);
            // }
            // printf("\n");
            #pragma endregion
        }
        // break;
        ++sweep;
        #pragma omp parallel
        {
            int gpuid = omp_get_thread_num();
            cudaSetDevice(gpuid);
            if(gpuid == 0){
                judgeFunc<<<batch, 1024,0,stream1>>>(dev_allpass, dev_pass, p);   // concentrate each block's result(converged or not)
            }
            else{
                judgeFunc<<<batch, 1024,0,stream2>>>(dev_allpass_1, dev_pass_1, p1);   // concentrate each block's result(converged or not)
            }
        }
        cudaMemcpyAsync(host_allpass, dev_allpass, sizeof(unsigned) * batch, cudaMemcpyDeviceToHost,stream1);    
        cudaMemcpyAsync(host_allpass_1, dev_allpass_1, sizeof(unsigned) * batch, cudaMemcpyDeviceToHost,stream2);
        // printf("all pass\n");
        // for(int f = 0;f < batch;++f){
        //     printf("%d %d ",host_allpass[f],host_allpass_1[f]);
        //     // printf("%d ",host_allpass)
        // }
        // printf("\n");
        printf("now sweep %d\n",sweep);
        continue_flag = ((ifallpass(host_allpass, batch, p) && ifallpass(host_allpass_1,batch,p1)) || sweep>maxsweep);
    }
    dim3 dimGrid10(2 * p, batch, 1);
    dim3 dimBlock10(32, k, 1);
    #pragma omp parallel
    {
        int gpuid = omp_get_thread_num();
        cudaSetDevice(gpuid);
        if(gpuid == 0){
            getUDV<<<dimGrid10, dimBlock10,0,stream1>>>(dev_A, dev_U, dev_V, dev_V0, height, width_perdevice, height, width_perdevice, p, height/32, dev_diag, width_perdevice, k);  //&1.3
        }
        else{
            getUDV<<<dimGrid10, dimBlock10,0,stream2>>>(dev_A_1, dev_U_1, dev_V_1, dev_V1, height, width_perdevice, height, width_perdevice, p1, height/32, dev_diag_1, width_perdevice, k);  //&1.3
        }
    } 
    clock_t end = clock();
    printf("it cost %lf",(double)(end-begin)/CLOCKS_PER_SEC);
    printf("sweep:%d \n",sweep);

    double* host_diag = (double*)malloc(sizeof(double)*minmn*batch);
    double* host_diag1 = (double*)malloc(sizeof(double)*minmn*batch);
    cudaSetDevice(gpu0);
    cudaMemcpy(host_diag,dev_diag,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    cudaSetDevice(gpu0);
    cudaMemcpy(host_diag1,dev_diag_1,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    FILE* file2 = fopen("dev_diag.txt","w");
    for(int f = 0;f < minmn;++f){
        fprintf(file2,"%lf %lf ",host_diag[f],host_diag1[f]);
    }
    // double* host_U = (double*)malloc(sizeof(double)*height*height*batch);
    // cudaMemcpy(host_U,dev_U,sizeof(double)*height*height*batch,cudaMemcpyDeviceToHost);
    // FILE* file = fopen("dev_U.txt","w");
    // for(int f = 0;f < height;++f){
    //     for(int g=0;g<height;++g){
    //         fprintf(file,"%lf ",host_U[f*height+g]);
    //     }
    //     fprintf(file,"\n");
    // }
    // double* host_diag = (double*)malloc(sizeof(double)*minmn*batch);
    // double* host_diag1 = (double*)malloc(sizeof(double)*minmn*batch);
    // cudaMemcpy(host_diag,dev_diag,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_diag1,dev_diag_1,sizeof(double)*minmn*batch,cudaMemcpyDeviceToHost);
    // FILE* file2 = fopen("dev_diag.txt","w");
    // for(int f = 0;f < minmn;++f){
    //     fprintf(file2,"%lf %lf ",host_diag[f],host_diag1[f]);
    // }
    
    // printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

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