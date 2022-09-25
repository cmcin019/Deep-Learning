# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114


# Imports
import math
import random

# TODO: Question 01
# x is a vector of dim K
# A, B, C are square mats of dim K x K

def mat_vec(A, x) :
	K = len(x)
	mat=[]
	for i in range(K):
		result = 0
		for j in range(K):
			result += A[j][i] * x[j]
		mat.append(result)
	return mat


def sigmoid(y):
	sig = []
	for i in range(len(y)):
		sig.append(1 / (1 + math.exp( -y[i] )))
	return sig

def mat_add(u, v):
	res = []
	for i in range(len(u)):
		res.append(u[i] + v[i])
	return res
	
def eucld_dist(w):
	dist = sum([math.pow(i, 2) for i in w])
	return dist
	
	
def run():
	# √x2+z2+y2+w2
	# x√x2+z2+y2+w2

def main() -> None :
	pass
	

if __name__ == "__main__":
	main()
	print("End")






