###### MAIN SCRIPT FOR INTERPOLATION, FEATURE ARITHMETIC AND UI ######


## Module form loading the model and generating images ##

#import all libraries
import os
import keras
import shutil
import random
import imageio
import numpy as np
from numpy import load
import tensorflow as tf
import plotly.express as px
from matplotlib import pyplot
from numpy.random import randn
from keras.models import Model
from numpy import mean, expand_dims
from matplotlib import pyplot as plt
from keras.preprocessing import image
from tensorflow.keras import applications
from numpy import asarray, linspace, save
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


#function to interpolate between two latent points p1 and p2
def interpolate_points(p1, p2, steps=10):
     # interpolate ratios between the points
     ratios = linspace(0, 1, num=steps)
     # linear interpolate vectors
     vectors = list()
     for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
     return asarray(vectors)

#function to generate latent points randomly in the latent space.
def generate_latent_points(latent_dim, n_samples, n_classes=8):
 	random.seed(1010)
 	x_input = randn(latent_dim * n_samples)
 	z_input = x_input.reshape(n_samples, latent_dim) #Reshape to be provided as input to the generator.
 	return z_input

#to save multiple images in all folders
def image_transition(images, n, new_path,i,j):
  names = []
  
  os.chdir(str(i)+"__"+str(j))
  for x in range(n):
    pyplot.axis('off')
    pyplot.imshow(images[x, :, :])
    name = "img"+str(x)+".png"
    pyplot.savefig(name,bbox_inches='tight')
    print("saving ", name)
    names.append(name)
  return names

#Create GIFs from images
def transition_creation(image_name, full_path,i,j):
  #Open each file in the folder and join them together to make the gif  
  with imageio.get_writer(os.path.join(full_path,str(i)+"__"+str(j)+'.gif'), mode='I') as writer:
      print
      for filename in image_name:
          os.chdir(full_path)
          
          image = imageio.imread(filename)
          #appending the images
          writer.append_data(image)
      #Message to assure gif is made corresponding to the images    
      print("done making gif",i, "__",j)

#make new folders to keep the images into them initially
def make_dir(len_lp, path):
  
  for i in range(0, len_lp):
    j = 0
    if i==j:
      j=1
   
    dir = os.path.join(path, str(i)+"__"+str(j))
    if os.path.exists(dir):
      shutil.rmtree(dir)
    os.makedirs(str(i)+"__"+str(j))
        
#load the model
model = load_model('generator_model_128x128_350.h5')
# model = load('data.npy')
#generate latent points
latent_points = generate_latent_points(250, 25)
l = len(latent_points)
#saving the generated latent points in a .npy file so that same latent points could be used again.
#to do so uncomment the model = load('data.npy') and comment the line above.
save('data.npy', latent_points)
#define paths for the root directory
path = 'C://Users//JUNU//Desktop//Dissertation//FLSK//'
#define paths for the folder to place the generated data in
path_data = 'C://Users//JUNU//Desktop//Dissertation//FLSK//cc'
#define paths for the static folder of the HTML web pages
path_static = 'C://Users//JUNU//Desktop//Dissertation//FLSK//static'
make_dir(l, path)

#########################################
c =0
for i in range(0, l):
    j = 0
    if i==j:
      j=1
    c+=1
    os.chdir(path)
    tr= interpolate_points(latent_points[i], latent_points[j], 1)
    interpolated = tr*1
    X = model.predict(interpolated)
    X = (X + 1) / 2.0
    zz= image_transition(X, len(interpolated), path, i,j)
    print("done", c)


#%%

##      Module for file transfer    ##

RootDir1 = path
TargetFolder = path_data

#copying all the flies to the cc folder
q = 0
for a in range (len(latent_points)):
  b=0
  if a==b:
    b=1
  folder = path+str(a)+"__"+str(b)
  
  for root, dir, files in os.walk(folder):
    for name in files:
       
      if name.startswith('img0'):
        os.rename(os.path.join(folder,"img0.png"), os.path.join(folder,"flower"+str(q)+".png"))
        q+=1

#checking if file exists and starts with flower, then copy
for roots, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        
        for name in files:
          print(name)
                      
          if name.startswith('flower'):
             
              # print("file present")
            try:
                SourceFolder = os.path.join(roots,name)
                shutil.copy2(SourceFolder, TargetFolder)
            except shutil.SameFileError:
                pass
              
              
#change names of all the files and name them numerically             
os.chdir(TargetFolder)
for i in range(len(latent_points)):
  #original file name  
  old_name = "flower"+str(i)+".png"
  #new file name
  new_name = str(i)+".png"
  #renaming files
  try:
      os.rename(old_name, new_name)
  except FileExistsError:
      os.remove(new_name)
      os.rename(old_name, new_name)

#placing the content in the static/HTML folder for the webpage- UI    
src_dir = path_data
# path to destination directory
dest_dir = path_static+'\\html\\'
# getting all the files in the source directory
files = os.listdir(src_dir)
#command to copy 
shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


#%%
os.chdir(path)

#%%

#this is the helper code to plot t-SNE plots. The original source for this code and for reference is https://nextjournal.com/ml4a
#The objective of this code is to extract features of the images and save a pickle file which will in turn help group 
#similar images together based on features and then plot a t-SNE graph

#

#load images
def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=mod.input_shape[1:3])
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Use imagenet for vgg16 for feature extraction
mod = applications.vgg16.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=mod.input, outputs=mod.get_layer("fc2").output)
images_path = path_data
#search for image extensions
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
#maximum number of images taken
max_num_images = 1000
#search for images
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

print("keeping %d images to analyze" % len(images))




features = []
for i, image_path in enumerate(images):
    img, x = load_image(image_path)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('finished extracting features for %d images' % len(images))
from sklearn.decomposition import PCA

#using pca
features = np.array(features)
pca = PCA(n_components=15)
pca.fit(features)
pca_features = pca.transform(features)
query_image_idx = int(len(images) * random.random())
img = keras.utils.load_img(images[query_image_idx])


from scipy.spatial import distance
import pickle
similar_idx = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[1:6]
# load all the similarity results as thumbnails of height 100
thumbs = []
for idx in idx_closest:
    img = keras.utils.load_img(images[idx])
    img = img.resize((int(img.width * 100 / img.height), 100))
    thumbs.append(img)

# concatenate the images into a single image
concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

# show the image
plt.figure(figsize = (16,12))
plt.imshow(concat_image)

#saving pickle file
pickle.dump([images, pca_features, pca], open(path+'feature_flowers.p', 'wb'))


import os
import random
import numpy as np
import json
import matplotlib.pyplot
from scipy.spatial import distance
import pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
#loading pickle file
images, pca_features, pca = pickle.load(open(path+'feature_flowers.p', 'rb'))

for img, f in list(zip(images, pca_features))[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(img, f[0], f[1], f[2], f[3]))

num_images_to_plot = 25

if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]
X = np.array(pca_features)
#t-SNE parameters
tsne = TSNE(n_components=2, learning_rate=100, perplexity=22).fit_transform(X)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

#dimensions of the graph
width = 3000
height = 3000
max_dim = 500

#plotting the figure
full_image = Image.new('RGBA', (width, height))
for img, x, y in zip(images, tx, ty):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

matplotlib.pyplot.figure(figsize = (20,16))
imshow(full_image)
#saving the figure
plt.savefig(path_static+'//plot.png',bbox_inches='tight')

#%%

## Module for UI ##


#import all libraries
from flask import Flask, render_template, request
import plotly.express as px
from pathlib import Path
from PIL import Image
import pickle
import numpy as np

#initialise the FLASK app
app = Flask(__name__)

#Setting the route
@app.route("/")
def express():  
    print("")
    
    #render template returns the first webpage to be displayed
    return render_template("graph.html")

#The function gets executed after the render template, using the post method 
@app.route('/predict', methods=['POST'])
def graph():
    #taking user input
    data1 = request.form['p1']
    data2 = request.form['p2']
    #Changing directory for images to be displayed
    os.chdir(path_static)
    #function calling for the GAN model to interpolate and featrue arithmetic
    generate_images(data1, data2)
    os.chdir(path)
    #render template returns the second webpage to be displayed
    return render_template('after.html')

#avoiding caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



#interpolating between points based on user input
def interpolate_point(p1, p2, n_steps=10):
     # interpolate ratios between the points
     ratios = linspace(0, 1, num=n_steps)
     # linear interpolate vectors
     vectors = list()
     for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
        
     return asarray(vectors)


#generating frames or images for the gif
def image_transition_2(images, n,i,j):
  names = []
  for x in range(n):
    pyplot.axis('off')
    pyplot.imshow(images[x, :, :])
    name = "img"+str(x)+".png"
    #saving figure to png
    pyplot.savefig(name,bbox_inches='tight')
    names.append(name)
  return names

#Create GIFs
def transition_creation_2(image_name,i,j):
    #going thorugh all the images from image_transition_2
  with imageio.get_writer(('morphing.gif'), mode='I') as writer:
     
      for filename in image_name:
          image = imageio.imread(filename)
          writer.append_data(image)
      


def average_points(points, ix):
  # retrieve required vectors corresponding to the selected images
  vectors = points[ix]
  # average the vectors
  avg_vector = mean(vectors, axis = 0)
  return avg_vector

#Function to take user input from the webpage and implement interpolation
def generate_images(i,j):
  #user inputs  
  i = int(i)
  j = int(j)
  #getting them in lists
  flower_1 = [i]
  flower_2 = [j]

  # average vectors for each class
  feature1 = average_points(latent_points, flower_1)
  feature2 = average_points(latent_points, flower_2)
  # Vector arithmetic....
  result_vector = expand_dims((feature1+feature2), 0)
  #passing the resultant vector throught the model for prediction
  result_image = model.predict(result_vector)

  #scale pixel values for plotting
  result_image = (result_image + 1) / 2.0
  plt.imshow(result_image[0])
  #saving the figure
  plt.savefig(path_static+'//add.png',bbox_inches='tight')
  
  if i!=j:
    tr= interpolate_point(latent_points[i], latent_points[j], 50)
    interpolated = tr*1
    X = model.predict(interpolated)
    X = (X + 1) / 2.0
    # names = image_transition(X, len(interpolated), i,j)
    # transition_creation(names)
    zz= image_transition_2(X, len(interpolated), i,j)
    transition_creation_2(zz,i,j)
  return None    

#main function to run FLASK APP
if __name__ == '__main__':
    
    app.debug = False
    app.run()
