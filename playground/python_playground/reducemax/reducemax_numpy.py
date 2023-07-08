import numpy as np

def reducemax(arr):
    if arr.ndim == 1:
        return arr.max()
    elif arr.ndim == 2:
        return arr.max(axis=1)
    else:
        return arr.max(axis=2)

# Example usage
arr_1d = np.array([1, 2, 5, 4, 3])
result_1d = reducemax(arr_1d)
print("1D:\n"+str(result_1d)+"\n")

arr_2d = np.array([[1, 2, 3],
                   [6, 5, 4],
                   [7, 9, 8]])
result_2d = reducemax(arr_2d)
print("2D:\n"+str(result_2d)+"\n")

arr_3d = np.array([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                   
                   [[10, 11, 12],
                    [15, 14, 13],
                    [16, 18, 17]]])
result_3d = reducemax(arr_3d)
print("3D:\n"+str(result_3d))
