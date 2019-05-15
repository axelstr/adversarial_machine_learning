# FMG attack

# This program contains the DeepFool_attack module which creates an instance that
# finds adversarial examples derived from the DeepFool  method implemented in
# different norms.

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time



class DeepFool_attack:
	"""
	Container for running adversarial attacks on a Keras network.

	:param path: path to h5 keras model file. Assumed to have an architetcure where
	             the last layer has a linear activation function, followed by a pure
	             normalizing layer, for example, softmax.

	:attribute model:                  keras model object, loaded from path
	:attribute nr_outpus:              nr of outputs of the model
	:attribute path:                   path to h5 keras model file


	:attribute logits_model            parallell keras model without the last pure activaton layer
	:attribute logits_input_gradients: list of gradients for each logit with respect to input

	"""

	def __init__(self, path):
		"""
		Creates a DeepFool instance.
		"""

		# Basic loading
		self.model = tf.keras.models.load_model(path)
		self.nr_outputs = self.model.output.shape[1]
		self.path = path

		# Setting up gradients
		self.logits_input_gradients = [None] * self.nr_outputs

		self.logits_model = tf.keras.models.Model(inputs = self.model.input,
			outputs = self.model.layers[-2].output)

		for i in range(self.nr_outputs):
			myGradient = tf.keras.backend.gradients(self.model.layers[-2].output[0, i], self.model.input)
			self.logits_input_gradients[i] =  tf.keras.backend.function([self.model.input], myGradient)

	def attack(self, image, correct_label, norm = 2, step_size = 1, max_iter = 2000, overshoot = 0.02,
		clipping = None,  image_min = 0, image_max = 1,
		normalize = False, verbose = False, calc_imdiffs = False, target_label = None):
		"""
		Performs the DeepFool-attack on a single image.

		Algorithm described in Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
		DeepFool: a simple and accurate method to fool deep neural networks, https://arxiv.org/abs/1511.04599

		:param image:         numpy array of format H x W containing an image that can be fed to the model
		:param label:         Correct label of the image. Not used in the algorithm, only for getting more debugging output.label
		:param norm:          The l_p-norm, in which to do the attack.
		:param step_size:     The step length
		:param max_iter:      The maximum number of iterations before breaking
		:param overshoot:     The overshoot. At each iteration, the perturbation is multiplied with 1+0.02
		:param clipping:      None, "each" or "last"
		:param image_min:     Minimum value to be clipped to.
		:param image_max:     Maximum value to be clipped to.

		:returns adv_image:   numpy array of format H x W containing the adversarial image
		:returns success:     True if attack flipped original label, false otherwise
		:returns iteration:   the number of iterations needed to reach label flipping

		"""
		# Initialize output image
		adv_image = image

		# Initialize loop variables
		current_label = self.model.predict(np.array([image])).argmax()
		if verbose and current_label != correct_label: print("Input not correct")
		iteration = 0
		original_label = current_label
		w = np.zeros(image.shape)
		r_tot = np.zeros(image.shape)
		imdiffs = []

		# Calculate norm to use, see article on page 5
		q = 0
		if norm == np.inf:
			q = 1
		elif norm == 1 or norm == 0:
			q = np.inf
		else:
			q = (norm)/(norm - 1)

		# Break conditions for targeted or untargeted attacks
		if target_label != None:
			def cont():
				return current_label != target_label
		else:
			def cont():
				return current_label == original_label

		old_adv_image = image
		alterable_pixels = np.ones(np.shape(adv_image))
		while cont():
			if verbose: print(iteration)
			if iteration > max_iter:
				break

			# Calc gradients, logits and setup pert
			grad = self.logits_input_gradients[original_label]([np.array([adv_image])])[0][0, ...]
			logit = self.logits_model.predict(np.array([adv_image]))[0][original_label]
			pert = np.inf

			# If targeted, calculate gradients for only that target
			if target_label:
				if target_label == original_label:
					break

				grad_target = self.logits_input_gradients[target_label]([np.array([adv_image])])[0][0, ...]
				w = grad_target - grad

				logit_target = self.logits_model.predict(np.array([adv_image]))[0][target_label]
				f_target = logit_target - logit

				# Calculate the perturbation
				# For numerical stability, add 0.00001
				f = (abs(f_target) + 0.00001) ##############

			# If not targeted, loop over all outputs
			else:
				for k in range(0, self.nr_outputs):
					if k == original_label:
						continue

					grad_k = self.logits_input_gradients[k]([np.array([adv_image])])[0][0, ...]
					w_k = grad_k - grad

					logit_k = self.logits_model.predict(np.array([adv_image]))[0][k]
					f_k = logit_k - logit

					# Calculate the perturbation
					# For numerical stability, add 0.00001
					pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten(), ord = q) ##############

					if pert_k < pert:
						pert = pert_k
						w = w_k
						f = abs(f_k) + 0.00001

			# Calculate optimal perturbations
			if norm == 0:
				ones = np.ones(np.shape(adv_image))
				impact = ((ones+np.sign(w))/2 - adv_image) * w
				if iteration > 0:
					alterable_pixels[argmax] = alterable_pixels[argmax]*.9
					impact = impact*alterable_pixels
				argmax = np.unravel_index(np.argmax(np.abs(impact)), np.shape(adv_image))
				r_i = np.zeros(np.shape(adv_image))
				r_i[argmax] = np.sign(w[argmax])

			elif norm == 1:
				max_idx = adv_image != image_max
				min_idx = adv_image != image_min

				# w_temp is all pixels that can be changed, that is, in the adv_image, if 1, it cannot be increased
				# and if 0, it cannot be decreased
				w_temp = ((w > 0)  * w) * max_idx + ((w < 0) * w) * min_idx
				idx = (w == np.max(np.abs(w_temp))) | (w == - np.max(np.abs(w_temp)))
				r_i = np.nan_to_num(f / w * idx)

			else:
				r_i = f * np.abs(w) ** (q - 1) *  np.sign(w) / (np.linalg.norm(w.flatten(), ord = q) ** q) #################################

			# If applicable, normalize perturbation
			if normalize:
				if norm == np.inf:
					r_i = np.sign(r_i)
				else:
					r_i = r_i / np.linalg.norm(r_i.flatten(), ord = norm)

			# Update image and label
			if clipping == 'each':
				adv_image = np.clip(step_size * r_i * (1 + overshoot) + adv_image, image_min, image_max)
			else:
				adv_image = step_size * r_i * (1 + overshoot) + adv_image
			current_label = self.model.predict(np.array([adv_image])).argmax()

			if calc_imdiffs: imdiffs.append(np.linalg.norm(adv_image.flatten() - old_adv_image.flatten(), ord = norm))

			# Update loop variabled
			old_adv_image = adv_image
			iteration = iteration + 1

			if verbose and iteration == max_iter: print("Maximum iterations reached", max_iter)

		if clipping == 'last':
			adv_image = np.clip(adv_image, image_min, image_max)
			current_label = self.model.predict(np.array([adv_image])).argmax()

		success = original_label != current_label

		# Verbose output
		if verbose: print("Total iterations: ", iteration)
		if verbose and current_label == original_label:
			print("Output not changed")

		if verbose and current_label == correct_label:
			print("Output now correct label")

		return adv_image, original_label, current_label, iteration, imdiffs
