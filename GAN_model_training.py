#This is the code for generating a general or vanilla GAN. The main source for this code is 
#https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/?unapproved=679656&moderation-hash=89059f832ffb6c9ba16b068829e44ea3#comment-679656
#changes are made to the code as and when required

#import all libraries
from numpy import zeros, ones
from numpy.random import randn, randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt

# define the standalone discriminator model
# Input would be 128x128x3 images and the output would be a binary (using sigmoid)

def define_discriminator(in_shape=(128,128,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(128, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 64x64
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 32x32
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 16x16
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 8x8
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

#Verify the model summary
test_discr = define_discriminator()
#print summary of the model
print(test_discr.summary())
#save plot
plot_model(test_discr, to_file='disc_model.png', show_shapes=True)

# define the standalone generator model
# Generator must generate 128x128x3 images that can be fed into the discriminator. 
# So, we start with enough nodes in the dense layer that can be gradually upscaled
#to 128x128x3. 

def define_generator(latent_dim):
	model = Sequential()
	# Define number of nodes that can be gradually reshaped and upscaled to 128x128x3
	n_nodes = 128 * 8 * 8 #8192 nodes
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((8, 8, 128)))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 64x64
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 128x128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer 128x128x3
	model.add(Conv2D(3, (8,8), activation='tanh', padding='same')) #tanh goes from [-1,1]
	return model

test_gen = define_generator(250)
#print summary of the model
print(test_gen.summary())
#save plot
plot_model(test_gen, to_file='generator_model.png', show_shapes=True)

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

test_gan = define_gan(test_gen, test_discr)
#print summary of the model
print(test_gan.summary())
#save plot
plot_model(test_gan, to_file='combined_model.png', show_shapes=True)


# Function to sample some random real images
def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, 1)) # Class labels for real images are 1
	return X, y

# Function to generate random latent points
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	
	return x_input

# Function to generate fake images using latent vectors
def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples) #Generate latent points as input to the generator
	X = g_model.predict(x_input) #Use the generator to generate fake images
	y = zeros((n_samples, 1)) # Class labels for fake images are 0
	return X, y

# Function to save Plots after every n number of epochs
def save_plot(examples, epoch, n=10):
	# scale images from [-1,1] to [0,1] so we can plot
	examples = (examples + 1) / 2.0
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i])
	# save plot to a file so we can view how generated images evolved over epochs
	filename = 'generated_plot_128x128_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()

# Function to summarize performance periodically. 
# 
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=300):
	# Fetch real images
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real images - get accuracy
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# Generate fake images
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake images - get accuracy
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# Print discriminate accuracies on ral and fake images. 
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save generated images periodically using the save_plot function
	save_plot(x_fake, epoch)
	# save the generator model
	filename = 'generator_model_128x128_%03d.h5' % (epoch+1)
	g_model.save(filename)
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	plt.subplot(2, 1, 1)
	plt.plot(d1_hist, label='d-real')
	plt.plot(d2_hist, label='d-fake')
	plt.plot(g_hist, label='gen')
	plt.legend()
	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(a1_hist, label='acc-real')
	plt.plot(a2_hist, label='acc-fake')
	plt.legend()
	# save plot to file
	plt.savefig('results_baseline/plot_line_plot_loss.png')
	plt.close()
# train the generator and discriminator by enumerating batches and epochs. 
#
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2) #Disc. trained on half batch real and half batch fake images
	d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
	#  enumerate epochs
	for i in range(n_epochs):
		# enumerate batches 
		for j in range(bat_per_epo):
			# Fetch random 'real' images
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# Train the discriminator using real images
			d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' images 
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# Train the discriminator using fake images
			d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
			# Generate latent vectors as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# Label generated (fake) mages as 1 to fool the discriminator 
			y_gan = ones((n_batch, 1))
			# Train the generator (via the discriminator's error)
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# Report disc. and gen losses. 

			d1_hist.append(d_loss1)
			d2_hist.append(d_loss2)
			g_hist.append(g_loss)
			a1_hist.append(d_acc1)
			a2_hist.append(d_acc2)
		# evaluate the model performance, sometimes
		print('Epoch>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
								(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
		plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

############################################

#import necessary libraries
import os
import numpy as np
import cv2
from PIL import Image
import random

#Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/flowers/
#This is the dataset used for training the model

#Set number of images for training.
n=9540 
#set image size
SIZE = 128 
#Enter path to the dataset
path_to_data = 'ENTER PATH TO DATASET'
all_img_list = os.listdir(path_to_data) 

dataset_list = random.sample(all_img_list, n) #Get n random images from the directory

#Read all images
#image pre-processing
dataset = []
for img in dataset_list:
    #Read the images
    temp_img = cv2.imread(path_to_data + img)
    #opencv reads images as BGR so let us convert back to RGB
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) 
    #convert to array
    temp_img = Image.fromarray(temp_img)
    #Resize the images
    temp_img = temp_img.resize((SIZE, SIZE))
    #add it to the list
    dataset.append(np.array(temp_img))   

dataset = np.array(dataset) #Convert the list to numpy array

#Rescale to [-1, 1] -the generator uses tanh activation that goes from -1,1
dataset = dataset.astype('float32')
# scale to [-1,1] as dataset is [0,255]
dataset = (dataset - 127.5) / 127.5

# define the size of the latent space
latent_dim = 250
# create discriminator 
d_model = define_discriminator()
# create generator 
g_model = define_generator(latent_dim)
# combine the two to create the GAN
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300)