'''
Aditi Nair (asn264)
April 13 2017

Assignment 10, Problem 1

This is an MPI program in which even rank processes print "Hello from process -INSERT_RANK_NUMBER-" and odd rank 
processes print "Hello from process -INSERT_RANK_NUMBER-"

Run this program with Python3 and using the command:
mpiexec -n X python mpi_assignment_1.python

- where X is any positive integer.
'''

from mpi4py import MPI

#define MPI communicator
comm = MPI.COMM_WORLD

#get rank of current process
rank = comm.Get_rank()

#if rank is even, print 'Hello ...' message
if rank % 2 == 0:
	print ('Hello from process ' + str(rank))

#if rank is odd, print 'Goodbye ...' message
else:
	print ('Goodbye from process ' + str(rank))

