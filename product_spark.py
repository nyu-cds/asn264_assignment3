'''
Aditi Nair
May 7 2017

Assignment 3, Problem 2

This program creates an RDD containing the numbers from 1 to 1000, 
and then uses the fold method and mul operator to multiply them all together.
'''

from pyspark import SparkContext
from operator import mul

def main():

	#Create instance of SparkContext
	sc = SparkContext("local", "product")

	#Create RDD on the list of numbers [1,2,...,1000]
	nums = sc.parallelize(range(1,1000+1))

	#Use fold to aggregate data set elements by multiplicaton (ie multiply them all together)
	print(nums.fold(1,mul))

if __name__ == '__main__':
	main()

