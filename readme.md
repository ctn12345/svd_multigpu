## Third-Party Licenses

This project includes code from the following open-source projects:

- **[Project Name]**  
  - Original Author: MOLOjl  
  - License: BSD 3-Clause License  
  - Source: (https://github.com/MOLOjl/WCycleSVD/?tab=BSD-3-Clause-1-ov-file)  

The original license terms are included in the `LICENSE-THIRD-PARTY.txt` file.

# 算法步骤
我认为这次多GPU最主要要做好的工作就是任务分配该如何确保负载均衡，并且尽可能让SVD算法能在一定次数内收敛。
我觉得On Parallel Implementation of the One-sided Jacobi Algorithm for Singular Value Decompositions这篇论文给了我很大的启发，它写着在每一次sweep之前我们都需要对现已有的数据进行reordering，reordering成最开始数据大小序列。

下面我举一个例子就很容易清楚比如如下是一个序列它的值分别是 9 2 1 7。这四个值它们对应的是它们大小的排序序列应该是3 1 0 2。那么在每一次sweep完之后，我们的RoundRobin都应该先以最开始的序列索引3 1 0 2为开始，这样进行数据变换所需要的sweep更少。以上就是我们SVD计算每一次sweep之前需要做的准备工作

在进行矩阵旋转前我首先要有两个RoundRobin序列，它们一个的长度是2*p-1,一个长度是p;其中第一个2*p-1的就是长度为2*p的完整RoundRobin序列。我们可以设它为pab

而第二个单个长度为p的就对RoundRobin进行简单的裁剪了。
比如p=2 那么就是0 1 2 3 这样一个2*p的简单序列。我们首先设定第一次序列是0 2 1 3，然后我们固定住偶数序列，奇数序列向前移p-1次，这样就得到了2*p*(p-1)的序列了，设它为 pab1

所以目前的算法步骤是
step 1: 计算每一个k*height块矩阵norm和的大小，然后得到raw_order序列

step 2:将矩阵前一半部分分配给GPU1,矩阵后一半部分分配给GPU2。然后再进行one side jacobi变换一共要变换2*p-1次，利用的是pab的RoundRobin

step 3:将GPU1的后半部分数据和GPU2的后半部分数据进行交换，然后再进行one side jacobi 变换p次，利用pab1的RoundRobin  0 1 2 3--->0 3 2 1

step 4:将GPU1的后半部分数据和GPU2的前半部分数据进行交换，然后再进行one side jacobi变换p次，利用pab1的RoundRobin  0 3 2 1-->0 2 3 1

step 5:判断是否收敛，如果未收敛且sweep数目小于设定的阈值maxsweep数目就,那么就在进行reorder，并且jump到step2进行运算。



# 开始命令 测试部分
```
    nvcc -g -G ourtest2-24-success.cu -o singletest -lcusolver -lcublas
    ./singletest
```
# 开始命令 stream部分
```
    nvcc -g -G -Xcompiler -fopenmp  ourtest.cu -o test  -lcusolver -lcublas
    ./test
```
# 分析nsight system命令
```
    nsys profile -o my_system_report ./your_cuda_program -lnvToolsExt
```

# 测试多GPU代码
```
    nvcc  -g -G -Xcompiler -fopenmp paralle_sort_once.cu -o test -lcusolver -lcublas 
    ./test
```


