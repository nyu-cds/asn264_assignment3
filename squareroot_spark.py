'''
Aditi Nair
May 7 2017

Assignment 3, Problem 3 (BONUS)

Calculate the average of the square root of all the numbers from 1 to 1000.
'''

from pyspark import SparkContext
from math import sqrt
from operator import add

def main():

	#Create instance of SparkContext
	sc = SparkContext("local", "avg_square_root")

	#Create RDD on the list of numbers [1,2,...,1000]
	nums = sc.parallelize(range(1,1000+1))

	#Use map to compute square root of each value
	square_root_nums = nums.map(sqrt)

	#Sum all square roots using fold
	sum_square_roots = square_root_nums.fold(0,add)

	#Count number of values
	num_vals = (nums.count())

	#Print the sum of square roots divided by the number of values
	print (sum_square_roots/num_vals)

if __name__=='__main__':
	main()