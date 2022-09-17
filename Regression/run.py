# Author    : Cristopher McIntyre Garcia 
# Email     : cmcin019@uottawa.ca
# S-N       : 300025114

#################################################################################################
# Polynomial regression model on an x-y pair dataset of real valued random variables            #
# X takes value in (0, 1) and Y depends on X according to Y = cos(2πX) + Z                      #
# Z is a zero mean Gaussian random variable with variance σ^2 , and Z is independent of X       #
# Assuming the dependency between X and Y is unknow, and only a batch of N pairs is known       #
# Learn a polynomial regression model and examine the fitting and generalization capabilities   #
#################################################################################################

# Imports
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from typing import List, Tuple
import numpy as np

# TODO: (A) Write a function getData that generates a dataset {(xi , yi ) : i = 1, 2, ... N } of N (X, Y) pairs for a given value of N and σ^2.
def getData(
	size:int, 
	sigma:float
	) -> List[Tuple[float]] :

	# Generate list of data
	data = []
	for _ in range(size) : 
	
		# Generate instance 
		xi = random.uniform(0, 1) # Uniform distribution 
		zi = random.gauss(0, math.pow(sigma, 2)) # Gauss distribution with mean=0 and var=sigma
		yi = math.cos(2 * math.pi * xi) + zi # Values generated according to Y = cos(2πX) + Z
		
		# Add real x-y pair to dataset
		data.append((xi, yi))
		
	return data


# TODO: (B) Write a function getMSE which computes the mean square error (MSE) for a given dataset fitted to a specified polynomial.
def getMSE(
	dataset:List[Tuple[float]], 
	coefficients:List[float]
	) -> float :
	
	# MSE for a give dataset and coefficients
	mse = 0
	for (x, y) in dataset: # Computing the MSE on all the data in dataset
		
		# Compute predicted value for instance of x-y pair
		predicted = 0
		for i in range(len(coefficients)): # The lengthe of the coefficients is the degree of the polynomial
		
			# Adding coefficient-input pair to predicted output value 
			predicted += coefficients[i] * math.pow(x, i)
		
		# Squared difference of predicted and real outputs
		mse += math.pow(predicted - y, 2)
	
	# Final value of the MSE
	mse = mse / len(dataset)
	
	return mse


# TODO: (C) Write a function fitData that estimates the polynomial coefficients by fitting a given dataset to a degree-d polynomial.
def fitData(
	dataset:List[Tuple[float]], 
	degree:int, 
	sigma:float,
	coefficients=None,
	batch_size=1,
	epochs=50,
	lr=0.1,
	regularization_rate=0,
	plot_data=False
	) -> Tuple[List[float], float, float] :
	
	# Generate random coefficients if none are provided
	if coefficients == None :
		coefficients = [random.random() for _ in range(degree)]
	
	# Determine type of gradient decent from batch size
	if batch_size == 1: # Use SGD
	
		# Sample single random x-y pair from dataset
		samp_dataset = random.sample(dataset, 1)
		
	elif batch_size == len(dataset): # Use GD
	
		# Sample full dataset
		samp_dataset = dataset
		
	else: # Use Mini-Batched SGD
	
		# Sample random x-y pairs from dataset
		samp_dataset = random.sample(dataset, batch_size)
		
	E_in_plot = []
	E_out_plot = []
		
	# Iterate through epochs to fit coefficients to the dataset
	for epoch in range(epochs):
	
		new_coefficients = []
		for c_i in range(len(coefficients)): # Find partial derivative for each individual coefficient
			
			# Calculate gradient
			gradient = 0
			for (x, y) in samp_dataset: # Calculate gradient for each x-y pair in sample dataset
				summation = 0
				
				for i in range(len(coefficients)): 
					summation += coefficients[i] * math.pow(x, i)
					
				# Gradient value used for updating coefficient
				gradient += math.pow(x, c_i) * (y - summation)

			# Calculate new coefficient from gradient
			# If regularization rate = 0, no regularization is used
			regularization = regularization_rate * coefficients[c_i] 
			new_coefficients.append(coefficients[c_i] - lr * ((-2/batch_size * gradient) - regularization))
		
		# Update coefficients
		coefficients = new_coefficients
		
		# Computing Error
		# Error with fitted coefficients
		E_in= getMSE(dataset, coefficients)
		E_in_plot.append(E_in)
		
		# Generate new data and calculate error 
		new_data = getData(2000, sigma)
		E_out= getMSE(new_data, coefficients)
		E_out_plot.append(E_out)
		
		# For plotting and documentation
		if plot_data and math.fmod(epoch, 200) == 0.0:
			fig, ax = plt.subplots()
			ax.set(ylim=(-1, 1))
			plt.scatter(list(map(lambda data: data[0], dataset)), list(map(lambda data: data[1], dataset)))
			predicted_data =[]
			for (x, y) in dataset: # Computing the MSE on all the data in dataset
				# Compute predicted value for instance of x-y pair
				predicted = 0
				for i in range(len(coefficients)): # The lengthe of the coefficients is the degree of the polynomial
					# Adding coefficient-input pair to predicted output value 
					predicted += coefficients[i] * math.pow(x, i)
				predicted_data.append(predicted)
			plt.scatter(list(map(lambda data: data[0], dataset)), list(map(lambda data: data, predicted_data)))
			plt.title('Iteration' + str(epoch))
			plt.show()
			print(E_in, E_out)
		
	if plot_data:
		plt.title("Training Error")
		plt.plot(E_in_plot)
		plt.show()
		plt.title("Testing Error")
		plt.plot(E_out_plot)
		plt.show()
		fig, ax = plt.subplots()
		ax.set(xlim=(0, 300))
		plt.title("Testing Error Zoomed")
		plt.plot(E_out_plot)
		plt.show()
	# Error with fitted coefficients
	E_in = getMSE(dataset, coefficients)
	
	# Generate new data and calculate error 
	new_data = getData(2000, sigma)
	E_out = getMSE(new_data, coefficients)
	
	return (coefficients, E_in, E_out)


# TODO: (D) Experiment function 
def experiment(
	size:int, 
	degree:int, 
	sigma:float, 
	M=50, 
	algorithm='GD', 
	batch_size=1, 
	epochs=50,
	regularization_rate=0
	) -> Tuple[float, float, float] :
	
	# Determin batch size form gradient decent type
	# If algorithm is other, the provided batch size is used for Mini-Batched SGD
	if algorithm == 'GD':
		
		# Full dataset
		batch_size = size
		
	elif algorithm == 'SGD':
	
		# Single random value
		batch_size = 1

	# Run M experiments and collect averages
	E_in_av, E_out_av = 0, 0
	polynomials = []
	for _ in range(M):
		
		# Generate dataset and fit coefficients to data
		dataset = getData(size, sigma)
		coefficients, E_in, E_out = fitData(dataset, degree, sigma, epochs=epochs, batch_size=batch_size, regularization_rate=regularization_rate)
		
		# Sum of Errors
		E_in_av += E_in
		E_out_av += E_out
		
		# Final coefficients
		polynomials.append(coefficients)
		
	# Average Errors
	E_in_av /= M
	E_out_av /= M
	E_gen = (E_in_av + E_out_av) / 2
	
	# Average coefficients
	new_coefficients = [sum(i)/len(polynomials) for i in zip(*polynomials)]
	
	# Generate new dataset 
	new_data = getData(2000, sigma)
	
	# Error of average coefficients on new dataset
	E_bias = getMSE(new_data, new_coefficients)
	
	return E_in_av, E_out_av, E_bias, E_gen


# TODO: (E) Run experiment on different combinations of dataset size, polynomial degree, and variance
def run(include_gen=True, epochs=2, regularize=False) -> None :

	# Test different sizes, polynomial degrees, and variance
	N = [2, 5, 10, 20, 50, 100, 200]
	#N = [2, 5, 10]
	ds = [x for x in range(21)] 
	sigmas = [0.01, 0.1, 1.]
	
	reg_rate=0
	reg = ''
	if regularize :
		reg_rate = 0.001
		reg = '_regularized'
	
	anchor_x = .8
	if include_gen :
		anchor_x = .9

	print("SIZE IMPACT")
	for sig in sigmas:
		for d in ds:
			E_in_plot = []
			E_out_plot = []
			E_bias_plot = []
			E_gen_plot = []
			
			fig, ax = plt.subplots()
			ax.set(ylim=(0, 2))
			#fig = plt.figure(figsize=(10,5))
			for n in tqdm(N):
				E_in_av, E_out_av, E_bias, E_gen = experiment(n, d, sig, epochs=epochs, regularization_rate=reg_rate)
				E_in_plot.append(E_in_av)
				E_out_plot.append(E_out_av)
				E_bias_plot.append(E_bias)
				E_gen_plot.append(E_gen)
			
			plt.plot(N, E_in_plot, label ='E_in')
			plt.plot(N, E_out_plot, label ='E_out')
			plt.plot(N, E_bias_plot, label ='E_bias')
			if include_gen :
				plt.plot(N, E_gen_plot, label ='E_gen')
			plt.legend(bbox_to_anchor =(anchor_x, 1.15), ncol = 4)
			
			plt.xlabel("Batch size")
			plt.ylabel("MSE")
			
			plt.title('degree: ' + str(d) + ", sigma: " + str(sig))
			fig.savefig('N' + reg + '/'+'degree: ' + str(d) + ", sigma: " + str(sig) + ".jpg", bbox_inches='tight', dpi=150)
			plt.close()
		#plt.show()
		
	print("VARIANCE IMPACT")
	for n in N:
		for d in tqdm(ds):
			E_in_plot = []
			E_out_plot = []
			E_bias_plot = []
			E_gen_plot = []
			
			fig, ax = plt.subplots()
			ax.set(ylim=(0, 2))
			for sig in sigmas:
				E_in_av, E_out_av, E_bias, E_gen = experiment(n, d, sig, epochs=epochs, regularization_rate=reg_rate)
				E_in_plot.append(E_in_av)
				E_out_plot.append(E_out_av)
				E_bias_plot.append(E_bias)
				E_gen_plot.append(E_gen)
			
			plt.plot(sigmas, E_in_plot, label ='E_in')
			plt.plot(sigmas, E_out_plot, label ='E_out')
			plt.plot(sigmas, E_bias_plot, label ='E_bias')
			if include_gen :
				plt.plot(sigmas, E_gen_plot, label ='E_gen')
			plt.legend(bbox_to_anchor =(anchor_x, 1.15), ncol = 4)
			
			plt.xlabel("Variance")
			plt.ylabel("MSE")
			
			plt.title('degree: ' + str(d) + ", size: " + str(n))
			fig.savefig('Sigma' + reg + '/'+'degree: ' + str(d) + ", size: " + str(n) + ".jpg", bbox_inches='tight', dpi=150)
			plt.close()
		#plt.show()
		
	print("DEGREE IMPACT")
	for sig in sigmas:
		for n in tqdm(N):
			E_in_plot = []
			E_out_plot = []
			E_bias_plot = []
			E_gen_plot = []
			
			fig, ax = plt.subplots()
			ax.set(ylim=(0, 2))
			for d in ds:
				E_in_av, E_out_av, E_bias, E_gen = experiment(n, d, sig, epochs=epochs, regularization_rate=reg_rate)
				E_in_plot.append(E_in_av)
				E_out_plot.append(E_out_av)
				E_bias_plot.append(E_bias)
				E_gen_plot.append(E_gen)
			
			plt.plot(ds, E_in_plot, label ='E_in')
			plt.plot(ds, E_out_plot, label ='E_out')
			plt.plot(ds, E_bias_plot, label ='E_bias')
			if include_gen :
				plt.plot(ds, E_gen_plot, label ='E_gen')
			plt.legend(bbox_to_anchor =(anchor_x, 1.15), ncol = 4)
		
			plt.xlabel("Degree")
			plt.ylabel("MSE")
						
			plt.title('size: ' + str(n) + ", sigma: " + str(sig))
			fig.savefig('Degree' + reg + '/'+'size: ' + str(n) + ", sigma: " + str(sig) + ".jpg", bbox_inches='tight', dpi=150)
			plt.close()
		#plt.show()
		
		
	
	

# TODO: (F)   ###################################
              #### SEE ACCOMPANYING DOCUMENT ####
              ###################################

def main() -> None :
	sigma = .1
	data = getData(200, sigma)
	# mse = getMSE(data, [3.,2.,0.,2.,5.,4.])
	batch_size = len(data)
	degree = 5
	#(c, ei, eo) = fitData(data, degree, sigma, epochs=1, batch_size=batch_size)
	#print(ei, eo)
	
	(c, ei, eo) = fitData(data, degree+15, sigma, epochs=1000, batch_size=batch_size, plot_data=True)
	print(ei, eo)
	print()
	(c, ei, eo) = fitData(data, degree, sigma, epochs=1000, batch_size=batch_size, plot_data=True)
	print(ei, eo)
	print()
	(c, ei, eo) = fitData(data, degree, sigma, epochs=1000, batch_size=batch_size, plot_data=True, regularization_rate=.001)
	print(ei, eo)
	
	#(c, ei, eo) = fitData(data, degree, sigma, coefficients=c, epochs=500, batch_size=batch_size)
	#print(ei, eo)
	
	#E_in_av, E_out_av, E_bias, E_gen = experiment(200, degree, sigma, epochs=1000)
	#print(E_in_av, E_out_av, E_bias)
	
	#run()
	#print()
	#run(regularize=True)
	

if __name__ == "__main__":
	for _ in range(1):
		main()
		print()
	print("End")






