'''
Aditi Nair (asn264)
April 23 2017

UnitTests for parallel_sorter.
Also, parallel_sorter script includes an assert statement which will fail if program 
does not correctly sort all of the original values.

'''

import unittest
from parallel_sorter import *

class TestParallelSorter(unittest.TestCase):

	def test_is_sorted(self):
		
		#Test whether is_sorted function recognizes sorted and un-sorted functions
		self.assertTrue(is_sorted(np.arange(5)))
		self.assertFalse(is_sorted(np.array([2,3,1,4,0])))

	def test_partitioner_splits(self):

		#Correct partitioner will partition permutated np.arange(20) into 4 bins of equal size 5
		nums = np.random.permutation(np.arange(20))
		partitions = partition_nums(nums,20,4)
		for partition in partitions:
			self.assertTrue(len(partition)==5)

	def test_partitioner_losses(self):

		#Correct partitioner will not lose any values in the partitioning process
		nums = np.random.permutation(np.arange(20))
		partitions = partition_nums(nums,20,4)
		self.assertTrue(np.all(np.sort(np.hstack(partitions))==np.arange(20)))

if __name__ == '__main__':
	unittest.main()