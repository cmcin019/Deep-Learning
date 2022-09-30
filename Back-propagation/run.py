# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

# Imports
import math
from random import random

# TODO: Question 01

def mat_vec(A, x):
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

def vec_add(u, v):
	res = []
	for i in range(len(u)):
		res.append(u[i] + v[i])
	return res
	
def forward_loss(w):
	dist = sum([math.pow(i, 2) for i in w])
	return dist
	
def forward(x, A, B, C):
	net = {}
	net['x'] = x
	net['A'] = A
	net['B'] = B
	net['C'] = C
	net['y'] = mat_vec(A, x)
	net['v'] = mat_vec(B, x)
	net['u'] = sigmoid(net['y'])
	net['z'] = vec_add(net['u'], net['v'])
	net['w'] = mat_vec(C, net['z'])
	return net

def a_L_minus_1(net):
	C = net['C']
	gradients = []
	for i in range(len(C)):
		gradient = []
		for j in range(len(C[i])):
			gradient.append(2 * net['w'][j] * net['C'][i][j])
		gradients.append(gradient)
	return gradients

def C_gradient(net):
	C = net['C']
	gradients = []
	for i in range(len(C)):
		gradient = []
		for j in range(len(C[i])):
			gradient.append(2 * net['w'][j] * net['z'][i])
		gradients.append(gradient)
	return gradients

def B_gradient(net):
	B = net['B']
	gradients = []
	for i in range(len(B)):
		gradient = []
		for j in range(len(B[i])):
			gradient.append(net['x'][i] * sum(a_L_minus_1(net)[j]))
		gradients.append(gradient)
	return gradients

def A_gradient(net):
	A = net['A']
	gradients = []
	for i in range(len(A)):
		gradient = []
		for j in range(len(A[i])):
			gradient.append(net['x'][i] * sigmoid_derivation(net['y'][j]) * sum(a_L_minus_1(net)[j]))
		gradients.append(gradient)
	return gradients

def backward(f):
	C_g = C_gradient(f)
	B_g = B_gradient(f)
	A_g = A_gradient(f)
	return A_g, B_g, C_g

def gradient_descent(lr, N, net):
	K = len(net['x'])
	for _ in range(N):
		net = forward([random() for _ in range(K)], net['A'], net['B'], net['C'])
		A_g, B_g, C_g = backward(net)
		for i in range(K):
			for j in range(K):
				net['A'][i][j] = net['A'][i][j] - lr * A_g[i][j] / N
				net['B'][i][j] = net['B'][i][j] - lr * B_g[i][j] / N
				net['C'][i][j] = net['C'][i][j] - lr * C_g[i][j] / N
	return net
	
import torch
def test(x, A, B, C):
	t_x = torch.tensor(x, requires_grad=False)
	t_A = torch.tensor(A, requires_grad=True)
	t_B = torch.tensor(B, requires_grad=True)
	t_C = torch.tensor(C, requires_grad=True)

	net_forward = torch.matmul((torch.sigmoid(torch.matmul(t_x, t_A)) + torch.matmul(t_x, t_B)), t_C)

	loss = torch.sum(torch.pow(net_forward, 2))
	print("TORCH:")
	print("Torch loss", loss)
	loss.backward()
	print(f"{t_C.grad}\n")
	print(f"{t_B.grad}\n")
	print(f"{t_A.grad}\n\n")
	f = forward(x, A, B, C)
	A_g, B_g, C_g = backward(f)
	loss = forward_loss(f['w'])
	print("MINE:")
	print("My loss", loss)
	print(torch.tensor(C_g),'\n')
	print(torch.tensor(B_g),'\n')
	print(torch.tensor(A_g),'\n')

def run(K=3, N=1000, lr=0.01, perform_test=True):

	A = [[random() for _ in range(K)] for _ in range(K)]
	B = [[random() for _ in range(K)] for _ in range(K)]
	C = [[random() for _ in range(K)] for _ in range(K)]

	x = [random() for _ in range(K)]

	net = forward(x, A, B, C)
	# A_g, B_g, C_g = backward(f)
	
	# TEST #
	if perform_test:
		test(x, A, B, C)

	loss = forward_loss(net['w'])
	print("Loss Before GD:")
	print(loss)
	print(net['A'])
	print(net['B'])
	print(net['C'],'\n')
	net = gradient_descent(lr, N, net)
	loss = forward_loss(net['w'])
	print("Loss After GD:")
	print(loss)
	print(net['A'])
	print(net['B'])
	print(net['C'])

def main() -> None :
	run()

if __name__ == "__main__":
	main()
	print("End")

