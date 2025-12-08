from random import shuffle

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

## Lesson 4 - Arithmetic - Dec 7, 2025
print("\nLesson 4: Arithmetic\n")

arithmetic_array = np.array([1, 2, 3])

# Scalar arithmetic
print("Sum to all of the elements in the array", arithmetic_array + 1)
print("Substract to all of the elements in the array", arithmetic_array - 5)
print("Multiply to all of the elements in the array", arithmetic_array * 3)
print("Divide to all of the elements in the array", arithmetic_array / 2)

# Vectorized math functions
print("Square for each element", np.sqrt(arithmetic_array))
print("Round for each element", np.ceil(arithmetic_array))

# Constants
print("PI", np.pi)
print("Area of circles", np.pi*arithmetic_array**2)

# Element-wise arithmetic
arithmetic_array2 = np.array([4, 5, 6])
print("Sum of arrays", arithmetic_array + arithmetic_array2)

# Comparison operators
scores = np.array([70, 85, 100, 60, 75, 54, 23])
print("Failed students", scores <= 60)
scores[scores < 60] = 0
print("Asign 0 to failed students", scores)

## Lesson 5 - Broadcasting - Dec 7, 2025
print("\nLesson 5: Broadcasting\n")

# Broadcasting is a way numpy handles arithmetic operations between arrays of different shapes
# Numpy will virtually expand the smaller array to fit the bigger one
broadcasting_array1 = np.array([[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12],
                                [13, 14, 15, 16]])
broadcasting_array2 = np.array([[1], [2], [3], [4]])

print("Array 1 Shape", broadcasting_array1.shape)
print("Array 2 Shape", broadcasting_array2.shape)

print("Multiplication", broadcasting_array1 * broadcasting_array2)

## Lesson 6 - Aggregate functions - Dec 7, 2025
# Aggregate functions summarize the data in an array into a single value normally
print("\nLesson 6: Aggregate functions\n")
aggregate_array = np.array([[1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10]])

print("Sum of all elements", np.sum(aggregate_array))               # Sum of all elements
print("Mean og all elements", np.mean(aggregate_array))              # Mean of all elements
print("Standard deviation of all elements", np.std(aggregate_array))  # Standard deviation of all elements
print("Variance of all elements", np.var(aggregate_array))            # Variance of all elements
print("Minimum value", np.min(aggregate_array))                     # Minimum value
print("Maximum value", np.max(aggregate_array))                     # Maximum value
print("Index of minimum value", np.argmin(aggregate_array))         # Index of minimum value
print("Index of maximum value", np.argmax(aggregate_array))         # Index of maximum value
print("Sum columns", np.sum(aggregate_array, axis=0))        # Sum of each column
print("Sum rows", np.sum(aggregate_array, axis=1))           # Sum of each row

## Lesson 7 - Filtering arrays - Dec 7, 2025
# Selecting elements from an array based on conditions (like in JS)
print("\nLesson 7: Filtering arrays\n")
ages = np.array([[25, 30, 18, 45, 50],
                 [ 40, 15, 50, 20, 16]])

# Boolean indexing
print("Teenage students", ages[ages < 20])  # Teenage students
print("Adult students", ages[ages >= 20])    # Adult students
print("Even ages", ages[ages % 2 == 0])        # Even ages
print("Odd ages", ages[ages % 2 != 0])         # Odd ages

# Where function - Mantains the original shape of the array
# Takes 3 arguments: condition, array, value if false
adults = np.where(ages >= 18, ages, 0)
print("Ages with only adults", adults)

## Lesson 8 - Random Numbers - Dec 7, 2025
print("\nLesson 8: Random Numbers\n")

# Random Number Generator Object
# You can add a seed so you always get the same numbers, like in Minecraft
rng = np.random.default_rng(seed=2)

# First argument is inclusive, last exclusive
print("Random integer", rng.integers(low=1, high=101, size=(3, 2)))

# Floating numbers
print("Random 0-1 float", np.random.uniform())
print("Random -1-1 float", np.random.uniform(-1, 1, (3, 2)))

# Shuffle array
shuffle_array = np.array([1, 2, 3, 4, 5])
rng.shuffle(shuffle_array)
random_number = rng.choice(shuffle_array)
print("Shuffled array", shuffle_array)
print("Random number from array", random_number)