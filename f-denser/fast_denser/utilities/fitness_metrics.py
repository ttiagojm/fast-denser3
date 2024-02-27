# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import tensorflow as tf
from keras import backend as K
from fast_denser.utils import calc_K, empirical_ntk_jacobian_contraction, calc_eigen

def accuracy(y_true, y_pred):
	"""
	    Computes the accuracy.


	    Parameters
	    ----------
	    y_true : np.array
	        array of right labels
	    
	    y_pred : np.array
	        array of class confidences for each instance
	    

	    Returns
	    -------
	    accuracy : float
	    	accuracy value
    """


	y_pred_labels = np.argmax(y_pred, axis=1)

	return accuracy_score(y_true, y_pred_labels)



def mse(y_true, y_pred):
	"""
	    Computes the mean squared error (MSE).


	    Parameters
	    ----------
	    y_true : np.array
	        array of right outputs
	    
	    y_pred : np.array
	        array of predicted outputs
	    

	    Returns
	    -------
	    mse : float
	    	mean squared errr
    """

	return mean_squared_error(y_true, y_pred)

def relu_determinant(model, data):

	if isinstance(data, tuple):
		data = data[0]

	K_mat = tf.zeros((data.shape[0], data.shape[0]))

	relu_layers, layer_outputs = list(), list()

	for i, layer in enumerate(model.layers):
		layer_outputs.append(layer.output)

		if hasattr(layer, "activation") and layer.activation.__name__ == "relu":
			relu_layers.append(i)

	activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
	outputs = activation_model(data, training=True)

	# Select only ReLU layers
	for i, layer in enumerate(outputs):
		if i in relu_layers:
			K_mat = calc_K(K_mat, layer)

	return K_mat


def ntk(model, x1, x2):
	result = empirical_ntk_jacobian_contraction(
		model, 
		x1,
		x2
	)

	max_eig, min_eig = calc_eigen(result)

	return max_eig/min_eig