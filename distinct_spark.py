'''
Aditi Nair
May 7 2017

Assignment 3, Problem 1

Counts the number of distinct words in the input text.
'''

from pyspark import SparkContext
import re

def splitter(line):

	'''
	remove any non-words and split lines into separate words
	finally, convert all words to lowercase
	'''

	line = str(re.sub(r'^\W+|\W+$', '', line))
	return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':

	#Create instance of SparkContext
	sc = SparkContext("local", "distinct_word_count")
	
	#Read text file and create an RDD of strings, one for each line in the file
	text = sc.textFile('pg2701.txt')

	#Split lines and get a list of words
	words = text.flatMap(splitter)

	#Perform mapping step
	words_mapped = words.map(lambda x: (x,1))

	#distinct() returns a new dataset of distinct elements (key, value pairs) from words_mapped
	#Then count() returns returns the number of elements in the new dataset of distinct elements
	print (words_mapped.distinct().count())

