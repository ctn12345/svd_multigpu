import numpy as np 
# load diag data

# 读取矩阵文件
def read_matrix_from_file(file_path):
    # 加载矩阵数据，假设矩阵以空格分隔
    return np.loadtxt(file_path)

# 进行 SVD 分解
def perform_svd(matrix):
    U, S, Vt = np.linalg.svd(matrix)
    return U, S, Vt

size_1 = 128
file_path = "./data/generated_matrixes/A_h"+str(size_1)+"_w"+str(size_1)+".txt"

# 读取矩阵
matrix_all = read_matrix_from_file(file_path)
# print("length of matrix")
# print(len(matrix_all))
# batch = len(matrix_all)/(size_1*size_1)
# batch = int(batch)
batch = 2
print("batch "+str(batch))
dev_V_all = np.loadtxt("dev_V.txt")
dev_V_all = dev_V_all.reshape(1,-1)
dev_U_all = np.loadtxt("dev_U.txt")
dev_U_all = dev_U_all.reshape((1,-1))
# print(dev_U_all.shape)
diag_arr = np.loadtxt('dev_diag.txt')
diag_arr = diag_arr.reshape((1,-1))
dev_U = np.zeros((1,size_1*size_1))
dev_V = np.zeros((1,size_1*size_1))
dev_diag = np.zeros((1,size_1))
for number in range(batch):
    # begin = size_1*size_1*number
    # end = size_1*size_1*(number+1)
    matrix = matrix_all
    # print(matrix.shape)
    # break
    matrix = matrix.reshape((size_1,size_1))
    
    U, S, Vt = perform_svd(matrix.T)

    begin_diag = size_1*number
    end_diag = size_1*(number+1)
    dev_diag = diag_arr[:,begin_diag:end_diag]
    index_arr = np.argsort(-dev_diag)
    dev_diag = dev_diag[0,index_arr]
    # break
    # dev_diag = list(dev_diag)
    # index_number = list((range(len(dev_diag))))
    # index_diag_val = []
    # for i,j in zip(index_number,dev_diag):
    #     index_diag_val.append([i,j])
    # print(index_diag_val)
    # sorted_id = sorted(index_diag_val,key=lambda x : x[1],reverse=True)
    begin_V = size_1*size_1*number
    end_V = size_1*size_1*(number+1)
    # print(begin_V)
    # print(end_V)
    # print(dev_V_all.shape)
    dev_V = dev_V_all[0,begin_V:end_V]
    # print(dev_V.shape)
    dev_V = dev_V.reshape((size_1,size_1))
    
    begin_U = size_1*size_1*number
    end_U = size_1*size_1*(number+1)
    # print((dev_U_all[begin_U:end_U]).shape)
    dev_U = dev_U_all[:,begin_U:end_U]
    dev_U = dev_U.reshape((size_1,size_1))
    dev_diag = np.array(dev_diag)
    dev_U = np.array(dev_U)
    dev_V = np.array(dev_V)
    dev_U = dev_U[index_arr]
    dev_U = dev_U[0]
    dev_V = dev_V[index_arr]
    dev_V = dev_V[0]
    # print(dev_V.shape)
    # print(dev_U.shape)
    # print(dev_diag.shape)
    # break
    print("raw matrix")
    print(U*S@Vt)
    print("my result matrix")
    print(dev_U.T*dev_diag@dev_V)

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("U python")
    print(U)
    print("U my result")
    print(dev_U.T)

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")

    print("V python")
    print(Vt)
    print("V my result")
    print(dev_V)

