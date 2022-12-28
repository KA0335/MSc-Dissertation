###############################################################################
############### PYTHON SCRIPT FOR H5 FILE EVALUATION ##########################
###############################################################################

### ONLY USE IF YOU NEED TO CHECK THE H5 files. PLACE THEM IN THE RIGHT FOLDER####


#import all libraries
from numpy import asarray
from numpy.random import randn
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from brisque import BRISQUE
from numpy import mean, expand_dims


# Function to generate random latent points
def generate_latent_points(latent_dim, n_samples):
	ran_data = randn(latent_dim * n_samples)
    #For input to the generator reshaping the matrice
	output_data = ran_data.reshape(n_samples, latent_dim) 
	return output_data

# Function to create a plot of generated imagesx
def plot_generated(examples, n):
	# plot images
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :])
#to get the array in shape        
def average_points(points, ix):
 	#since indexing starts from 0, 1 is sibtracted 
 	zeroes = [i-1 for i in ix]
 	# retrieve required vectors corresponding to the selected images
 	vec = points[zeroes]
 	# average the vectors
 	avg_vec = mean(vec, axis = 0)
	
 	return avg_vec
	
#counter for epoch files starting from 10
c=10
#to store BRISQUE score and average BRISQUE score
dict_score = {}
dict_avg = {}

#Enter path to the h5 files
path_to_folder = 'C://Users//JUNU//Desktop//Dissertation//New folder//h5 1k epochs'
#path to the h5 files
path_to_h5 = path_to_folder+'//generator_model_128x128_' 
for i in range(40):
  
    if c <99:
# load the saved model
        model = load_model(path_to_h5+'0'+str(c)+'.h5')
    else:
        model = load_model(path_to_h5+str(c)+'.h5')
    #latent point generation
    latent_points = generate_latent_points(250, 25)
    # use loaded generator model to generate the image matrices
    X  = model.predict(latent_points)
    # scale from [-1,1] to [0,1] 
    X = (X + 1) / 2.0
    plot_generated(X, 5)
    # load model
    score_list = []
    for j in range(len(latent_points)):
        flower_1 = [j+1]
        # average vectors for each class
        feature1 = average_points(latent_points, flower_1)
        # Vector arithmetic....
        result_vector = expand_dims((feature1), 0)
        #passing the resultant vector throught the model for prediction
        result_image = model.predict(result_vector)
        # scale pixel values for plotting
        result_image = (result_image + 1) / 2.0
        #plotting each image
        plt.imshow(result_image[0])
        fig1 = plt.gcf()
        #showing each image
        plt.show()
        plt.draw()
        #shutting off the axis
        plt.axis('off')
        #saving to a jpg file according to epoch and image number
        fig1.savefig(str(c)+"_"+str(j)+'.jpg',bbox_inches='tight',transparent=True, pad_inches=0)

        #calculating BRISQUE SCORE for each image
        name = str(c)+"_"+str(j)+'.jpg'
        obj = BRISQUE(name, url=False)
        points = obj.score()
        
        score_list.append(points)
        
    
    score_list.pop(0)
    mean_list = mean(score_list)
   
    #Adding BRISQUE score and mean score from all images to 2 different Dictionaries
    dict_score[c] = score_list
    dict_avg[c] = mean_list
    print("Done epoch", c)
    c+=10




#%%
list_of_index = []
list_of_values = []
m = 100
mx = 0
for i,v in dict_score.items():
    list_of_index.append(i)
    list_of_values.append(v)
    if m>min(v):
        m = min(v)
        min_i = i
    if mx<max(v):
        mx = max(v)
        max_i =i

print("Minimum BRISQUE SCORE", m)
print("The imdex",min_i)
print("Maximum BRISQUE score",mx)
print("The index",max_i)





