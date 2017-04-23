'''
Aditi Nair (asn264)
April 23 2017

This is a parallelized sorting program using MPI. Run this program with Python3 and using the command:
< mpiexec -n X python parallel_sorter.py > - where X is # of processes.

The root process generates a large unsorted array and slices into bins by value.
It sends one bin to each process (including itself) using a Scatter operation, and each process sorts its bin. 
Once each process has sorted its bin, the root process uses a Gather operation to 
receive the sorted bins and combines them into a single array. 

EMBEDDED TEST: This script will raise an assertion error if the data was not properly processed and sorted.

'''

import numpy as np
from mpi4py import MPI

def partition_nums(nums,max_val,num_processes):

	'''
	We partition the array nums such that each process will be sent all values in nums that appear in a certain range.
	Each range is (about) the same size. The value bin_range is the number of values each process is responsible for sorting.

	Therefore bin[0] will contain values for process 0, and these values will be in the range [0,bin_range),
	bin[1] will contain values for process 1, and these values will be in the range [bin_range,2*bin_range), etc...

	If nums = np.random.permutation(np.arange(max_val)), then this function will assign each processer the same amount of work. 
	Since nums is drawn from a uniform distribution, this is a reasonable approximation of distributing the work evenly.
	'''

	bins = []
	bin_range = max_val/num_processes
	
	for rank in range(num_processes):
		
		lower_bound = rank*bin_range
		upper_bound = (rank+1)*bin_range
		
		if rank != num_processes - 1:
			bin = nums[(nums>=lower_bound)&(nums<upper_bound)]
		else:
			bin = nums[(nums>=lower_bound)]
		bins.append(bin)

	return bins

def is_sorted(nums):

	'''Checks if nums is sorted in ascending order.'''

	return np.all(nums[:-1] <= nums[1:])


#Define MPI communicator
comm = MPI.COMM_WORLD

#Get rank of current process
rank = comm.Get_rank()

#Get number of processes
num_processes = comm.Get_size()

num_vals = 10**4
max_val = 10**4

if rank == 0:

	#Generate a large un-sorted array of size num_vals, integer values uniformly drawn from [0,max_vals)
	nums = np.random.randint(0,max_val,num_vals)

	#Print nums before sort
	print ('Nums before sort: ', nums)

	#Partition the nums for broadcasting - see function for details
	bins = partition_nums(nums,max_val,num_processes)

else:

	#Placeholder
	bins = None

#Scatter operation to send appropriate bin to appropriate processer
bin = comm.scatter(bins,root=0)

#Sort bin 
bin = np.sort(bin)

#Gather operation sends sorted bins to root
nums = comm.gather(bin,root=0)

#Root process appends sorted bins to get original nums in sorted order
if rank == 0:

	nums = np.hstack(nums)

	#Test that nums is sorted and that it has not lost any values 
	assert is_sorted(nums)&(len(nums)==num_vals)

	print ('Passed sorting test...')
	print ('Nums after sort: ', nums)


