import numpy as np

# 读取矩阵文件
def read_matrix_from_file(file_path):
    # 加载矩阵数据，假设矩阵以空格分隔
    return np.loadtxt(file_path)

# 进行 SVD 分解
def perform_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix)
    return U, S, Vt

# 主函数
if __name__ == "__main__":
    # 文件路径
    size_1 = 4096
    file_path = "./data/generated_matrixes/A_h"+str(size_1)+"_w"+str(size_1)+".txt"
    
    # 读取矩阵
    matrix = read_matrix_from_file(file_path)
    print(matrix)
    matrix = matrix.reshape((size_1,size_1))
    U, S, Vt = perform_svd(matrix)
    jointG = matrix @ Vt.T
    np.savetxt('jointG.txt', U, fmt='%f')
    print(S)
    # matrix = matrix.T
    # height = matrix.shape[0]
    # width = matrix.shape[1]
    # # print(height,width)
    # # print("原始矩阵：")
    # # print(matrix)
    # # print("矩阵相乘:")
    # part_mat1 = matrix[:32]
    # part_mat2 = matrix[32:]
    # part_joint1 = part_mat1.T@part_mat1
    # # print(part_joint1)
    # print("第一部分")
    # print(part_mat1)
    # print(part_mat1.shape)

    # # print("转置矩阵相乘")
    # # s = matrix.T@matrix
    # # print(s)
    
    # # SVD 分解
    # # matrix = matrix.T
    # U, S, Vt = perform_svd(part_mat1)
    # # U1,S1,Vt1 = perform_svd(matrix[32:])
    
    # # 输出结果
    # # print("\n左奇异矩阵 U：")
    # # print(U)
    # # print("\n奇异值 S：")
    # # print(S)
    # # np.savetxt()
    # # print("\n右奇异矩阵 Vt：")
    # # print(Vt)
    # np.savetxt('output.txt', S, fmt='%f')
    # # print("\n 奇异值 S2：")
    # # print(S1)
