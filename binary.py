'''
Aditi Nair (asn264)
February 24 2017

Assignment Four

zbits is a function that takes two arguments n and k and 
prints all binary strings of length n that contain k zero bits,
one per line.
'''

import itertools
import sys

class InvalidZbitsException(Exception):

	'''This is specifically for inputs n and k which do not satisfy 0<=k<=n.'''
	pass


def zbits(n,k):
	
	'''
	First we ensure the following inequality holds: 0 <= k <= n.
	If this is true, we create a "master string" which contains a single binary string of length n 
	with k zero bits - here we use ITERTOOLS.CHAIN.
	
	Next we use ITERTOOLS.PERMUTATIONS to get all possible permutations of characters in master_str.
	However, ITERTOOLS.PERMUTATIONS is position-based so master_str='00' will have two permutations, 
	not one. Therefore we apply the set function to the iterator before printing the results.
	'''

	if k<=n and k>=0: #check input
		master_str=''.join([i for i in itertools.chain(k*'0',(n-k)*'1')]) #create a "master string"
		for bin_str in set(itertools.permutations(master_str)): #generate all 'unique' permutations
			print ''.join(i for i in bin_str)
	else:
		raise InvalidZbitsException



def main():
	
	query = raw_input('Please enter two positive integers n and k, separated by a comma.\n')
	
	try:
		n,k = map(int,query.strip().split(','))
		zbits(n,k)
	except (ValueError,InvalidZbitsException):
		print 'Invalid input.'
		sys.exit()


if __name__ == '__main__':
	main()