# Examples

# This program uses the fgm_attack and deepfool_attack modulse to create four
# adversarial samples for each method in the four norms: 0,1,2,infinity.

# ----------------- Import libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(style = 'darkgrid')
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('text', usetex=True)

# Import attack module
from fgm_attack import FGM_attack
from deepfool_attack import DeepFool_attack

# ------------------ Load dataset

class Data_set():
    """
    Given the name of a data set
    - MNIST
    - fashion-MNIST
    - (or) CIFAR-10
    an instance is created with the following parameters:
    - train_data
    - train_labels
    - test_data
    - test_labels
    - class_names
    The data is given as numpy nd array of images (2d array) with each pixel
    value in [0,1]. With CIFAR-10 each pixel is given as RGB array of three
    float values in [0,1]. train_labels and test_labels are each arrays of
    integers that are the labels of the corresponding images in train_data,
    test_data.
    """
    def __init__(self, set):
        if set == 'MNIST':
            keras_dataset = keras.datasets.mnist
        elif set == 'fashion-MNIST':
            keras_dataset = keras.datasets.fashion_mnist
        elif set == 'CIFAR-10':
            keras_dataset = keras.datasets.cifar10
        else:
            assert False, "Please enter valid data set."

        (train_images, train_labels), (test_images, test_labels) = keras_dataset.load_data()

        self.train_data = train_images / 255.0
        self.test_data = test_images / 255.0
        if set == 'CIFAR-10':
            self.train_labels = [lbl[0] for lbl in train_labels]
            self.test_labels = [lbl[0] for lbl in test_labels]
        else: self.test_labels = test_labels

        class_names_dict = {
            'MNIST': ['0','1','2','3','4','5','6','7','8','9'],
            'fashion-MNIST': ['Top', 'Trousers', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
            'CIFAR-10': ['plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        }
        self.class_names = class_names_dict[set]

# ------------------- Main method

def generate_example_images():

    # Parameters after convenience
    target_label = None
    max_iter = 10000        # Maximum number of iterations

    # Instances
    mnist = Data_set('MNIST')
    cifar = Data_set('CIFAR-10')
    class_names = cifar.class_names
    fgm_mnist_attacker = FGM_attack('models/DNN_models/MNIST.h5')
    fgm_cifar_attacker = FGM_attack('models/CNN_models/CIFAR-10.h5')
    df_mnist_attacker = DeepFool_attack('models/DNN_models/MNIST.h5')
    df_cifar_attacker = DeepFool_attack('models/CNN_models/CIFAR-10.h5')

    # Computed data allocation
    images = []
    labels = []
    def store_image_data_mnist(x, x_adv, starting_label, adv_label):
        """Appends the current MNIST images and labels to images and labels
        lists."""
        images.append(x)                # original image
        images.append(np.abs(x_adv-x))  # perturbation
        images.append(x_adv)            # adversarial image
        labels.append(starting_label)   # label for original image
        labels.append('')               # unnamed perturbation
        labels.append(adv_label)        # label for adversarial image
    def store_image_data_cifar10(x, x_adv, starting_label, adv_label):
        """Appends the current CIFAR-10 images and labels to images and labels
        lists."""
        images.append(x)
        images.append(np.abs((x_adv-x))/np.linalg.norm(np.ndarray.flatten(np.abs((x_adv-x))), np.inf))
        images.append(x_adv)
        labels.append(cifar.class_names[starting_label])
        labels.append('')
        labels.append(cifar.class_names[adv_label])

    # Attacks
    # L0
    print('L0:')
    x = mnist.test_data[0]
    x_adv, starting_label, adv_label, steps = fgm_mnist_attacker.L0_attack(x,
                    target_label = target_label,
                    max_iter = max_iter,
                    avoid_jam = False,
                    verbose = False)
    print('L_0 IFGM complete in %s steps'%(steps))
    store_image_data_mnist(x, x_adv, starting_label, adv_label)
    x_adv, starting_label, adv_label, steps, _ = df_mnist_attacker.attack(x, correct_label = None,
            norm = 0, step_size = 1, max_iter = max_iter, overshoot = 0.02,
    		clipping = 'each',  image_min = 0, image_max = 1,
    		normalize = True, verbose = False, calc_imdiffs = False, target_label = None)
    print('L_0 DeepFool complete in %s steps'%(steps))
    store_image_data_mnist(x, x_adv, starting_label, adv_label)

    # L1
    print('L1:')
    x = cifar.test_data[14]
    x_adv, starting_label, adv_label, steps = fgm_cifar_attacker.L1_attack(x,
                    alpha = 1e-2,
                    target_label = target_label,
                    max_iter = max_iter,
                    verbose = True)
    print('L_1 IFGM complete in %s steps'%(steps))
    store_image_data_cifar10(x, x_adv, starting_label, adv_label)
    x_adv, starting_label, adv_label, steps, _ = df_cifar_attacker.attack(x, correct_label = None,
            norm = 1, step_size = 1e-2, max_iter = max_iter, overshoot = 0.02,
    		clipping = 'each',  image_min = 0, image_max = 1,
    		normalize = True, verbose = False, calc_imdiffs = False, target_label = None)
    print('L_1 DeepFool complete in %s steps'%(steps))
    store_image_data_cifar10(x, x_adv, starting_label, adv_label)

    # L2
    print('L2:')
    x = mnist.test_data[2]
    x_adv, starting_label, adv_label, steps = fgm_mnist_attacker.L2_attack(x,
                    alpha = 1e-3,
                    target_label = target_label,
                    max_iter = max_iter,
                    verbose = False)
    print('L_2 IFGM complete in %s steps'%(steps))
    store_image_data_mnist(x, x_adv, starting_label, adv_label)
    x_adv, starting_label, adv_label, steps, _ = df_mnist_attacker.attack(x, correct_label = None,
            norm = 2, step_size = 1e-3, max_iter = max_iter, overshoot = 0.02,
    		clipping = 'each',  image_min = 0, image_max = 1,
    		normalize = True, verbose = False, calc_imdiffs = False, target_label = None)
    print('L_2 DeepFool complete in %s steps'%(steps))
    store_image_data_mnist(x, x_adv, starting_label, adv_label)

    # Linf
    print('Linf:')
    x = cifar.test_data[8]
    x_adv, starting_label, adv_label, steps = fgm_cifar_attacker.Linf_attack(x,
                    alpha = 1e-4,
                    target_label = target_label,
                    max_iter = max_iter,
                    verbose = False)
    print('L_inf IFGM complete in %s steps'%(steps))
    store_image_data_cifar10(x, x_adv, starting_label, adv_label)
    x_adv, starting_label, adv_label, steps, _ = df_cifar_attacker.attack(x, correct_label = None,
            norm = np.inf, step_size = 1e-4, max_iter = max_iter, overshoot = 0.02,
    		clipping = 'each',  image_min = 0, image_max = 1,
    		normalize = True, verbose = False, calc_imdiffs = False, target_label = None)
    print('L_inf DeepFool complete in %s steps'%(steps))
    store_image_data_cifar10(x, x_adv, starting_label, adv_label)

    plot_example_image(images, labels)


def plot_example_image(images, labels):
    # Plot results
    norm_str = [r'$L_0$',r'$L_1$',r'$L_2$',r'$L_\infty$']

    n_rows = 4
    n_cols = 6
    fig, axes = plt.subplots(n_rows,n_cols, constrained_layout=True)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i][j].imshow(images[6*i+j],
                    cmap = plt.cm.binary,
                    aspect = 'equal',
                    # vmin = 0,
                    # vmax = 1,
                    )
            axes[i][j].set_xlabel(str(labels[6*i+j]))
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            axes[i][j].grid(False)
            if j == 0:
                axes[i][j].set_ylabel(norm_str[i], rotation = 90)
            if j == 1 or j == 4:
                axes[i][j].set_ylabel(r'$+$', rotation = 0)
            if j == 2 or j == 5:
                axes[i][j].set_ylabel(r'$=$', rotation = 0)
            if (i,j) == (0,1):
                axes[i][j].set_title('IFGM')
            if (i,j) == (0,4):
                axes[i][j].set_title('DeepFool')

    plt.savefig('_examples.png', dpi = 300)

if __name__ == '__main__':
    generate_example_images()
