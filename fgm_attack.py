# FMG attack

# This program contains the FGM_attack module which creates an instance that
# finds adversarial examples derived from the FGSM method implemented in
# different norms.


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

class FGM_attack():
    """
    Container for running Fast Gradient Method adversarial attacks on a
    Keras network.
    """

    # TODO: Implement target label method
    def __init__(self, model_path):
        """
        Creates FGM attack instance.
        """

        self.model = tf.keras.models.load_model(model_path)
        self.nr_outputs = self.model.output.shape[1]

        # For setting up FGSM
        self.loss_gradients = [None] * self.nr_outputs
        self.setupFGM()

    def setupFGM(self):
        """
        Initiates the instance by creating gradient methods with respects to
        to all outputs.
        """

        for i in range(self.nr_outputs):
            # Prepare
            categorical = tf.keras.utils.to_categorical(i, self.nr_outputs)
            pred_variable = tf.keras.backend.variable(categorical)

            # Create loss function
            loss = tf.keras.metrics.categorical_crossentropy(self.model.output, pred_variable)

            # Create gradients
            gradients = tf.keras.backend.gradients(loss, self.model.input)
            gradient_function = tf.keras.backend.function([self.model.input], gradients)
            self.loss_gradients[i] = gradient_function

    def L0_attack(self, image,
            target_label = None,
            max_iter = 10000,
            avoid_jam = True,
            verbose = False):
        """
        This method contiously alterns one pixel until misssclassification.

        The method uses the FGM method as a foundation. Each gradient value
        is multiplied with how much it's corresponding pixel could change
        before clipping. The pixel where a change to one or zero has largest
        impact is changed to that value. Pixels are iteratively changed
        until missclassification or target label is achieved.
        """

        # Prepare data and methods
        x = np.array([image])
        ones = np.ones(np.shape(x))
        starting_label = self.model.predict(x).argmax()
        current_label = starting_label
        if target_label != None:    # Prepare to reach target label
            dir = -1
            gradient_function = self.loss_gradients[target_label]
            def cont():
                return current_label != target_label
        else:               # Prepare to simply cause any missclassification
            dir = 1
            gradient_function = self.loss_gradients[starting_label]
            def cont():
                return current_label == starting_label
        if avoid_jam:
            alterable_pixels = np.ones(np.shape(x))
        if verbose: print('\tStarting label: \t',starting_label)

        # Iterate
        steps = 0
        while cont():
            steps += 1
            gradient_values = gradient_function([x])[0]*dir
            impact = ((ones+np.sign(gradient_values))/2 - x)*gradient_values
            if avoid_jam and steps > 1:
                alterable_pixels[argmax] = alterable_pixels[argmax]*.9
                impact = impact*alterable_pixels
            argmax = np.unravel_index(np.argmax(impact[0]), np.shape(x))
            x[argmax] = (np.sign(gradient_values[argmax])+1)/2
            current_label = self.model.predict(x).argmax()
            if steps == max_iter:
                x = ones
                if verbose: print('Unsuccesfull attack, breaking after %s  \
                                and returning empty image.'%(steps))
                break

        # Return results
        if verbose: print('\tAdversarial label: \t',current_label,'after',steps,'steps')
        return x[0], starting_label, current_label, steps

    def L1_attack(self, image, alpha,
            clipping = 'each',
            target_label = None,
            max_iter = 10000,
            verbose = False):
        """
        This function contiously adds/substracts alpha to the pixel
        corresponding with the highest absolute value in the gradient. This is
        repeated until missclassification.
        """

        # Prepare data and methods
        x = np.array([image])
        starting_label = self.model.predict(x).argmax()
        current_label = starting_label
        if target_label != None:    # Prepare to reach target label
            dir = -1
            gradient_function = self.loss_gradients[target_label]
            def cont():
                return current_label != target_label
        else:               # Prepare to simply cause any missclassification
            dir = 1
            gradient_function = self.loss_gradients[starting_label]
            def cont():
                return current_label == starting_label
        if verbose: print('\tStarting label: \t',starting_label)

        # Iterate
        steps = 0
        while cont():
            steps += 1
            gradient_values = gradient_function([x])[0]*dir
            if clipping == 'each':  # Max/min values can't be inc./decreased
                for index, value in np.ndenumerate(x):
                    if value == 0 and gradient_values[index]<0:
                        gradient_values[index] = 0
                    elif value == 1 and gradient_values[index]>0:
                        gradient_values[index] = 0
            argmax = np.unravel_index(np.argmax(np.abs(gradient_values[0])), np.shape(x))
            x[argmax] = x[argmax] + alpha*np.sign(gradient_values[argmax])
            if clipping == 'each':
                x = np.clip(x, 0, 1)
            current_label = self.model.predict(x).argmax()
            if steps == max_iter: break
        if clipping == 'last': x = np.clip(x, 0, 1)

        # Return results
        if verbose: print('\tAdversarial label: \t',current_label,'after',steps,'steps')
        return x[0], starting_label, current_label, steps

    def L2_attack(self, image, alpha, p_norm = 2,
            clipping = 'each',
            target_label = None,
            max_iter = 1000,
            verbose = True):
        """
        This method iteratively adds perturbs the image until missclassification.
        Each perturbation is calculated from the current gradient of the
        cost function. The gradient is then normalized in the given norm and
        multiplied with alpha before added to the adversarial perturbation.
        """

        # Prepare data and methods
        x = np.array([image])
        starting_label = self.model.predict(x).argmax()
        current_label = starting_label
        gradient_function = self.loss_gradients[starting_label]
        if target_label != None:
            dir = -1
            gradient_function = self.loss_gradients[target_label]
            def cont():
                return current_label != target_label
        else:
            dir = 1
            gradient_function = self.loss_gradients[starting_label]
            def cont():
                return current_label == starting_label
        if verbose: print('\tStarting label: \t',starting_label)

        # Iterate
        steps = 0
        while cont():
            steps += 1
            gradient_values = gradient_function([x])[0]*dir
            gradient_norm = np.linalg.norm(gradient_values[0].flatten(), 2)
            if gradient_norm == 0.0:
                print('gradient is ZERO, breaking attack')
                break
                # print(gradient_values)
                # x = x + alpha*np.sign(np.random.randn(1,np.shape(x)[1],np.shape(x)[2]))
            x = x + alpha * gradient_values/gradient_norm
            if clipping == 'each':
                x = np.clip(x, 0, 1)
            current_label = self.model.predict(x).argmax()
            if steps == max_iter: break

        # Return results
        if verbose: print('\tAdversarial label: \t',current_label,'after',steps,'steps')
        if clipping == 'last': x = np.clip(x, 0, 1)
        return x[0], starting_label, current_label, steps


    def Linf_attack(self, image, alpha,
            clipping = 'each',
            target_label = None,
            max_iter = 10000,
            verbose = True):
        """
        Performs a BIM attack on a single image.
        """

        # Prepare data and methods
        x = np.array([image])
        starting_label = self.model.predict(x).argmax()
        current_label = starting_label
        gradient_function = self.loss_gradients[starting_label]
        if target_label != None:
            dir = -1
            gradient_function = self.loss_gradients[target_label]
            def cont():
                return current_label != target_label
        else:
            dir = 1
            gradient_function = self.loss_gradients[starting_label]
            def cont():
                return current_label == starting_label
        if verbose: print('\tStarting label: \t',starting_label)

        # Iterate
        steps = 0
        while cont():
            steps += 1
            gradient_values = gradient_function([x])[0]*dir
            x = x + alpha * np.sign(gradient_values)
            if clipping == 'each':
                x = np.clip(x, 0, 1)
            current_label = self.model.predict(x).argmax()
            if steps == max_iter: break

        # Return results
        if verbose: print('\tAdversarial label: \t',current_label,'after',steps,'steps')
        if clipping == 'last': x = np.clip(x, 0, 1)
        return x[0], starting_label, current_label, steps
