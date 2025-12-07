import numpy as np

## Lesson 1 numpy arrays - Dec 6, 2025
print("\nLesson 1: Numpy arrays\n")

# Why numpy arrays are better
# 1. Performance: Numpy arrays are more efficient memory wise and faster than normal python lists
# 2. Functionality: Numpy provides a wide range of mathematical functions
# 3. Convenience: Numpy arrays support vectorized operations
# That's why everyone uses numpy for ml and data science

array = np.array([1, 2, 3, 4])
array = array * 2

print("Numpy array:", array)
print(type(array))

## Lesson 2 Multi-dimensional arrays - Dec 6, 2025
print("\nLesson 2: Multi-dimensional arrays\n")

# The number of elements in each dimension must be the same
zero_dementional_array = np.array("A")
print(zero_dementional_array.ndim, "dimensional array:", zero_dementional_array, "shape:", zero_dementional_array.shape)
one_dementional_array = np.array([1, 2, 3, 4])
print(one_dementional_array.ndim, "dimensional array:", one_dementional_array, "shape:", one_dementional_array.shape)
two_dementional_array = np.array([[1, 2, 3],
                                  [4, 5, 6]])
print(two_dementional_array.ndim, "dimensional array:", two_dementional_array, "shape:", two_dementional_array.shape)
three_dementional_array = np.array([[[1, 2, 3], [4, 5, 6]],
                                   [[7, 8, 9], [10, 11, 12]]])
print(three_dementional_array.ndim, "dimensional array:", three_dementional_array, "shape:", three_dementional_array.shape)

#Chain indexing
print("Accessing array w/chain indexing (default):", three_dementional_array[0][0][0])

#Multi-dimensional indexing -> Faster than chain indexing
print("Accessing array w/multidimensional indexing (faster np version):", three_dementional_array[0, 0, 0])

#Test
sum = three_dementional_array[0, 1, 2] + three_dementional_array[1, 1, 0]
print("The sum using multidimentional indexing is:", sum)

## Lesson 3 - Slicing - Dec 6, 2025
print("\nLesson 3: Slicing\n")

slicing_array = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]])

# When you're slicing an array using numpy yo use 3 arguments
# array[start:end:step]
# Starting index is inclusive, ending is exclusive
# Step can be negative
print("Slicing multidimensional array (row selection)")
print(slicing_array[0:3:2])

print("Slicing multidimensional array (column selection)")
print(slicing_array[:, 0:4:3])

print("Slicing multidimensional array (both column & row selection)")
print(slicing_array[0:2, 2:])