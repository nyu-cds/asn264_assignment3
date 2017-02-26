'''
Aditi Nair (asn264)
February 24 2017

Assignment Four

zbits is a function that takes two arguments n and k and 
prints all binary strings of length n that contain k zero bits,
one per line.
'''

import itertools

def zbits(n,k):
	
	'''
	First we ensure the following inequality holds (0 <= k <= n) and the values are ints.
	If this is true, we create a "master string" which contains a single binary string of length n 
	with k zero bits - here we use ITERTOOLS.CHAIN.
	
	Next we use ITERTOOLS.PERMUTATIONS to get all possible permutations of characters in master_str.
	However, ITERTOOLS.PERMUTATIONS is position-based so master_str='00' will have two permutations, 
	not one. Therefore we apply the set function to the iterator before returning the results.
	'''

	if isinstance(k,int) and isinstance(n,int) and k<=n and k>=0: #check input
		master_str=''.join([i for i in itertools.chain(k*'0',(n-k)*'1')]) #create a "master string"
		return {''.join(bin_str) for bin_str in itertools.permutations(master_str)} #generate all 'unique' permutations			
	else:
		return 'Invalid Input.'



def main():
	
	import binary
	
	assert binary.zbits(4, 3) == {'0100', '0001', '0010', '1000'}
	assert binary.zbits(4, 1) == {'0111', '1011', '1101', '1110'}
	assert binary.zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}



if __name__ == '__main__':
	main()