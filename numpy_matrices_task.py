"""
Assignment 2 - Understanding How Computers Hold Data (NumPy Basics)
--------------------------------------------------------------------
AI models don't "see" images or text directly — everything gets converted
into grids of numbers (matrices). This script demonstrates:
  1. Creating matrices (grids) using NumPy
  2. Performing basic math operations on them (add, subtract, multiply)
"""

import numpy as np

# ---------------------------------------------------------
# 1. Creating Grids (Matrices) of Numbers
# ---------------------------------------------------------

# A simple 2x3 matrix (2 rows, 3 columns) - like a tiny grayscale image
matrix_a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Another 2x3 matrix of the same shape
matrix_b = np.array([
    [6, 5, 4],
    [3, 2, 1]
])

print("Matrix A:")
print(matrix_a)
print("\nMatrix B:")
print(matrix_b)

# NumPy can also auto-generate matrices for us
zeros_matrix = np.zeros((2, 3))       # a grid full of zeros
ones_matrix = np.ones((2, 3))         # a grid full of ones
random_matrix = np.random.randint(0, 10, size=(2, 3))  # random ints 0-9
identity_matrix = np.eye(3)           # identity matrix (used a lot in AI math)

print("\nZeros Matrix:")
print(zeros_matrix)

print("\nOnes Matrix:")
print(ones_matrix)

print("\nRandom Matrix:")
print(random_matrix)

print("\nIdentity Matrix:")
print(identity_matrix)

# ---------------------------------------------------------
# 2. Basic Math on Matrices
# ---------------------------------------------------------

# Element-wise addition (adds numbers in the same position)
addition_result = matrix_a + matrix_b
print("\nA + B (element-wise addition):")
print(addition_result)

# Element-wise subtraction
subtraction_result = matrix_a - matrix_b
print("\nA - B (element-wise subtraction):")
print(subtraction_result)

# Element-wise multiplication (each cell multiplied by the matching cell)
elementwise_mult_result = matrix_a * matrix_b
print("\nA * B (element-wise multiplication):")
print(elementwise_mult_result)

# True matrix multiplication (dot product) - needs compatible shapes,
# so we multiply A (2x3) with B transposed (3x2) -> result is 2x2
dot_product_result = matrix_a.dot(matrix_b.T)
print("\nA . B^T (matrix / dot product multiplication):")
print(dot_product_result)

# Scalar operations (applying one number to the whole grid)
scaled_matrix = matrix_a * 10
print("\nMatrix A scaled by 10:")
print(scaled_matrix)

# ---------------------------------------------------------
# 3. Useful Grid Info (things AI models check constantly)
# ---------------------------------------------------------

print("\nShape of Matrix A:", matrix_a.shape)   # (rows, columns)
print("Data type of Matrix A:", matrix_a.dtype)
print("Total elements in Matrix A:", matrix_a.size)
