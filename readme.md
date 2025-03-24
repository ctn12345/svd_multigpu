## Third-Party Licenses

This project includes code from the following open-source projects:

- **[Project Name]**  
  - Original Author: MOLOjl  
  - License: BSD 3-Clause License  
  - Source: (https://github.com/MOLOjl/WCycleSVD/?tab=BSD-3-Clause-1-ov-file)  

The original license terms are included in the `LICENSE-THIRD-PARTY.txt` file.

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