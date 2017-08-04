from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		random.seed(1)

		self.synaptic_weights = 2* random.random((3,1)) - 1


	#the sigmoid function, which describes as ns shaped curve
	# we pass the weighted sum of the inputs through this function
	# to normalize them between 0 and 1
	#back propgation
	def __sigmoid(self,x):
		return 1/(1 + exp(-x))

	def __sigmoid_derivative(self,x):
		return x*(1-x)

    #meat
	def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
		for iteration in range(number_of_training_iterations):

			output = self.predict(training_set_inputs)
			error = training_set_outputs - output
			adjustments = dot(training_set_inputs.T , error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustments

	def predict(self,inputs):
		return self.__sigmoid(dot(inputs,self.synaptic_weights))



if __name__ == '__main__':

		#initialize a neural network
		neural_network = NeuralNetwork()

		print('Random starting synaptic weights:')
		print(neural_network.synaptic_weights)

		#the training set. We have 4 examples each consisting of 3 inputs
		# and 1 output
		training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,0,1]])
		training_set_outputs = array([[0,1,1,0]]).T

		#train
		# iterate 10k times and make small adjustments each time
		neural_network.train(training_set_inputs,training_set_outputs,10000)
		print('New synaptic weights after training:')
		print(neural_network.synaptic_weights)

		#test the neural network
		print('predicting:')
		print(neural_network.predict(array([1,0,0])))