# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

# Imports
import math
from random import random

# TODO: Question 01
# Matrix - vector multiplication used for weighted sumations
def mat_vec(A, x):
	K = len(x)
	mat=[]
	# A is a KxK matrix
	for i in range(K):
		result = 0
		for j in range(K):
			# Multiply and sum value to result
			result += A[j][i] * x[j]
		mat.append(result)
	return mat

# Perform sigmioid function on all elements of y
def sigmoid(y):
	sig = []
	for i in range(len(y)):
		# Append to resulting vector 
		sig.append(1 / (1 + math.exp( -y[i] )))
	return sig

# Derivative of the sigmoid function
def sigmoid_derivation(out):
	# Derivative of sigmoid used in backpropagation
	return 1 / (1 + math.exp( -out )) * (1 - 1 / (1 + math.exp( -out )))

# Element wise addition of two vectors 
def vec_add(u, v):
	res = []
	for i in range(len(u)): # vectros share the same length
		# Append to resulting vector 
		res.append(u[i] + v[i])
	return res
	
# Loss in our case is just the sum of all powers of a vector
def forward_loss(w):
	dist = sum([math.pow(i, 2) for i in w])
	return dist
	
# Forward creates a dictrionary with all the compositions of our functions
def forward(x, A, B, C):
	net = {}
	net['x'] = x # Input
	net['A'] = A # 'A' layer weights
	net['B'] = B # 'B' layer weights
	net['C'] = C # 'C' layer weights
	net['y'] = mat_vec(A, x) # Weighted sum of input with weigths in A
	net['v'] = mat_vec(B, x) # Weighted sum of input with weigths in B
	net['u'] = sigmoid(net['y']) # Sigmoid on all elemnts of y
	net['z'] = vec_add(net['u'], net['v']) # Element wise addition between u and v
	net['w'] = mat_vec(C, net['z']) # Fianl weighted sum of elements in z and weights in C
	# Network representation as a dictionary
	return net

# Compute the impact of the final layer on the loss
def a_L_minus_1(net):
	C = net['C']
	gradients = []
	for i in range(len(C)):
		gradient = []
		for j in range(len(C[i])):
			# Impact of initial weights on the loss
			gradient.append(2 * net['w'][j] * net['C'][i][j])
		gradients.append(gradient)
	return gradients

# Compute the gradientt of C
def C_gradient(net):
	C = net['C']
	gradients = []
	for i in range(len(C)):
		gradient = []
		for j in range(len(C[i])):
			# The gradient is the partial derivative of the loss function with respect to the output vector  * the partial derivative of the weighted sum with respect to the weights
			gradient.append(2 * net['w'][j] * net['z'][i])
		gradients.append(gradient)
	return gradients

def B_gradient(net):
	B = net['B']
	gradients = []
	for i in range(len(B)):
		gradient = []
		for j in range(len(B[i])):
			# The gradient is the partial derivative of the weighted sum with respect to the weights * the impact of the final weights
			gradient.append(net['x'][i] * sum(a_L_minus_1(net)[j]))
		gradients.append(gradient)
	return gradients

def A_gradient(net):
	A = net['A']
	gradients = []
	for i in range(len(A)):
		gradient = []
		for j in range(len(A[i])):
			# Similar to finding B plus adding the activation function derivative
			gradient.append(net['x'][i] * sigmoid_derivation(net['y'][j]) * sum(a_L_minus_1(net)[j]))
		gradients.append(gradient)
	return gradients

# Findind the impact of all the weights in our network
def backward(f):
	# Compute all gradients
	C_g = C_gradient(f)
	B_g = B_gradient(f)
	A_g = A_gradient(f)
	return A_g, B_g, C_g

def gradient_descent(lr, N, net): # Updating the weights of the network
	K = len(net['x'])
	for _ in range(N):
		net = forward([random() for _ in range(K)], net['A'], net['B'], net['C'])
		A_g, B_g, C_g = backward(net)
		# Update weights with gradients
		for i in range(K):
			for j in range(K):
				net['A'][i][j] = net['A'][i][j] - lr * A_g[i][j] / N
				net['B'][i][j] = net['B'][i][j] - lr * B_g[i][j] / N
				net['C'][i][j] = net['C'][i][j] - lr * C_g[i][j] / N
	return net
	
import torch
# Testing to see if gradients match those found with pytorch
def test(x, A, B, C):
	t_x = torch.tensor(x, requires_grad=False)
	t_A = torch.tensor(A, requires_grad=True)
	t_B = torch.tensor(B, requires_grad=True)
	t_C = torch.tensor(C, requires_grad=True)

	net_forward = torch.matmul((torch.sigmoid(torch.matmul(t_x, t_A)) + torch.matmul(t_x, t_B)), t_C)

	loss = torch.sum(torch.pow(net_forward, 2))
	print(f"x = \n{x}\n")
	print(f"A = \n{A}\n")
	print(f"B = \n{B}\n")
	print(f"C = \n{C}\n")
	print("TORCH:")
	print("Torch loss", loss)
	loss.backward()
	print(f"Gradient C = \n{t_C.grad}\n")
	print(f"Gradient B = \n{t_B.grad}\n")
	print(f"Gradient A = \n{t_A.grad}\n\n")
	f = forward(x, A, B, C)
	A_g, B_g, C_g = backward(f)
	loss = forward_loss(f['w'])
	print("MINE:")
	print("My loss", loss)
	# print(torch.tensor(C_g),'\n')
	# print(torch.tensor(B_g),'\n')
	# print(torch.tensor(A_g),'\n')
	print(f'Gradien C = \n{C_g}','\n')
	print(f'Gradien B = \n{B_g}','\n')
	print(f'Gradien A = \n{A_g}','\n')

def run(K=3, N=1000, lr=0.01, perform_test=True):
	# Generate random weights and variable
	A = [[random() for _ in range(K)] for _ in range(K)]
	B = [[random() for _ in range(K)] for _ in range(K)]
	C = [[random() for _ in range(K)] for _ in range(K)]

	x = [random() for _ in range(K)]

	# Forward through net, create net dictionary
	net = forward(x, A, B, C)
	
	# TEST #
	if perform_test:
		test(x, A, B, C)

	# Compute loss
	loss = forward_loss(net['w'])
	print("Loss Before GD:")
	print('Loss:', loss)
	print(f"A = \n{net['A']}")
	print(f"B = \n{net['B']}")
	print(f"C = \n{net['C']}",'\n')
	# Perform gradient descent 
	for _ in range(5): # epochs
		net = gradient_descent(lr, N, net)
	# Compute loss after gd to compare 
	loss = forward_loss(net['w'])
	print("Loss After GD:")
	print(loss)
	print(f"A = \n{net['A']}")
	print(f"B = \n{net['B']}")
	print(f"C = \n{net['C']}",'\n')

def main() -> None :
	run()

if __name__ == "__main__":
	main()
	print("End")

