import numpy as np

import test1



# 创建两个3x3矩阵
mat1 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

mat2 = np.array([[9, 8, 7],
                 [6, 5, 4],
                 [3, 2, 1]])

# 调用C++函数进行矩阵乘法
result = test1.testMatrixMultiplication()

print("Result of multiplication:")
print(result)
