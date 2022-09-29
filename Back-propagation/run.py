# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

# Imports
import math
from random import random

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

def sigmoid_derivation(out):
	return 1 / (1 + math.exp( -out )) * (1 - 1 / (1 + math.exp( -out )))

def mat_add(u, v):
	res = []
	for i in range(len(u)):
		res.append(u[i] + v[i])
	return res
	
def eucld_dist(w):
	dist = sum([math.pow(i, 2) for i in w])
	return dist
	
def forward(x, A, B, C):
	net = {}
	net['x'] = x
	net['y'] = mat_vec(A, x)
	net['u'] = mat_vec(B, x)
	net['v'] = sigmoid(net['y'])
	net['z'] = mat_add(net['u'], net['v'])
	net['w'] = mat_vec(C, net['z'])
	return net

def C_gradient(C, net):
	gradients = []
	for i in range(len(C)):
		gradient = []
		for j in range(len(C[i])):
			gradient.append(2 * net['w'][j] * net['z'][i])
		gradients.append(gradient)
	return gradients

def B_gradient(B, net, C_g):
	gradients = []
	for i in range(len(B)):
		gradient = []
		for j in range(len(B[i])):
			# gradient.append(4 * net['x'][i] * sum(C_g[j]) * (net['u'][j] + net['v'][j]))
			print(net['x'][i])
			print(net['u'][i])
			print(net['z'][i])
			print(net['w'][i])
			print(B)
			print(C_g)
			gradient.append(2*net['x'][i] * net['w'][j])
			gradient.append(4*sum(C_g[j]) * net['x'][i] * net['u'][i])
			# gradient.append(math.prod(C_g[j]) 	* net['x'][i] )
			# gradient.append(1 					* net['x'][i] )
			
			# gradient.append(sum(C_g[i]) 		* net['x'][i] )
			# gradient.append(math.prod(C_g[i]) 	* net['x'][i] )
		
		gradients.append(gradient)
	return gradients

def A_gradient(A, net, C_g):
	gradients = []
	for i in range(len(A)):
		gradient = []
		for j in range(len(A[i])):
			gradient.append(net['x'][i] * sigmoid_derivation(net['y'][i]) * math.prod(C_g[j]))
		gradients.append(gradient)
	return gradients

def backward(x, A, B, C):
	pass
	
import torch
def test(x, A, B, C):
	t_x = torch.tensor(x, requires_grad=False)
	t_A = torch.tensor(A, requires_grad=True)
	t_B = torch.tensor(B, requires_grad=True)
	t_C = torch.tensor(C, requires_grad=True)

	net_forward = torch.matmul((torch.sigmoid(torch.matmul(t_x, t_A)) + torch.matmul(t_x, t_B)), t_C)

	# external_grad = torch.tensor([1., 1.])
	# net_forward.backward()

	# loss = torch.nn.functional.mse_loss(net_forward, t_x)
	loss = torch.sum(torch.pow(net_forward, 2))
	loss.backward()

	print("REAL:")
	print(f"{t_C.grad}")
	print()
	print(f"{t_B.grad}")
	print()
	print(f"{t_A.grad}")
	print()

def run():

	K = 1
	A = [[random() for _ in range(K)] for _ in range(K)]
	B = [[random() for _ in range(K)] for _ in range(K)]
	C = [[random() for _ in range(K)] for _ in range(K)]

	x = [random() for _ in range(K)]

	f = forward(x, A, B, C)

	C_g = C_gradient(C, f)
	B_g = B_gradient(C,f,C_g)
	A_g = A_gradient(A,f,C_g)

	test(x, A, B, C)


	# print(f)
	# print()
	print(A,C)
	print("MINE:")
	print(torch.tensor(C_g))
	print()
	print(torch.tensor(B_g))
	print()
	print(torch.tensor(A_g))


	# gradient = []
	# i=0
	# for j in range(len(B[i])):
	# 	gradient.append(f['x'][i] * sum(C_g[j]))
	# 	print(gradient[-1])
	# 	gradient.append(f['x'][j] * sum(C_g[i]))
	# 	print(gradient[-1])
	# 	gradient.append(f['x'][i] * math.prod(C_g[j]))
	# 	print(gradient[-1])
	# 	gradient.append(f['x'][j] * math.prod(C_g[i]))
	# 	print(gradient[-1])


	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))
	# 	gradient.append(f['x'][i] * sum(f['x']) * sum(C_g[j]))




def main() -> None :
	run()
	

if __name__ == "__main__":
	main()
	print("End")

