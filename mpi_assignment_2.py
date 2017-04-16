'''
Aditi Nair (asn264)
April 13 2017

Assignment 10, Problem 2

In this program there is an arbitrary number of processes. 
Process 0 reads a value (val) from the user and verifies that it is an integer less than 100. 
Process 0 sends the value to Process 1 which multiplies it by its rank (which is 1).
Process 1 sends the value to Process 2 which multiplies it by its rank (which is 2).
Etc...
The last process sends the value back to process 0, which prints the result

Run this program with Python3 and using the command:
mpiexec -n X python mpi_assignment_2.python

- where X is any positive integer.
'''

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	
	#Read an integer below 100 from the user
	while True:
		try:
			val = int(input('\nEnter an integer below 100: '))
			if val < 100:
				break
		except ValueError:
			pass

	#Announce user-provided val
	print ('\nProcess ' + str(rank) + ' read user val ' + str(val))
	
	#Send to "next" process. If size=1, then val will be sent back to 0.
	comm.send(val, dest=(rank+1)%size)

	#Receive val from the "last" process with rank size-1.
	final_val = comm.recv(source=size-1)

	print ('Process ' + str(rank) + ' received val ' + str(final_val) + ' from Process ' + str(size-1))
	print ('\nFinal value is ' + str(final_val))

else:
	
	#Receive new val from previous process
	val = comm.recv(source=rank-1)

	#Announce received val
	print ('Process ' + str(rank) + ' received val ' + str(val) + ' from Process ' + str(rank-1))

	#Multiply val by rank number
	val *= rank

	#Send to "next" process. If this is the last process, then val will be sent back to 0. 
	comm.send(val, dest=(rank+1)%size)
	




