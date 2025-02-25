# 开始命令 测试部分
```
    nvcc -g -G ourtest2-24-success.cu -o singletest -lcusolver -lcublas
    ./singletest
```
# 开始命令 stream部分
```
    nvcc -g -G ourtest.cu -o test -lcusolver -lcublas
    ./test
```