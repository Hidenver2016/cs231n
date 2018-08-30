# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:20:19 2018

@author: hjiang
"""

# As usual, a bit of setup

import time, os, json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Introducing TinyImageNet
data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)

#TinyImageNet-100-A classes
for i, names in enumerate(data['class_names']):
  print (i, ' '.join('"%s"' % name for name in names))
  
# Print out all the keys and values from the data dictionary
for k, v in data.items():
  if type(v) == np.ndarray:
    print (k, type(v), v.shape, v.dtype)
  else:
    print (k, type(v))

#Visualize Examples
# Visualize some examples of the training data
classes_to_show = 7
examples_per_class = 5

class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)
for i, class_idx in enumerate(class_idxs):
  train_idxs, = np.nonzero(data['y_train'] == class_idx)
  train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)
  for j, train_idx in enumerate(train_idxs):
    img = deprocess_image(data['X_train'][train_idx], data['mean_image'])
    plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)
    if j == 0:
      plt.title(data['class_names'][class_idx][0])
    plt.imshow(img)
    plt.gca().axis('off')

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Pretrained model
model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')

#Pretrained model performance

batch_size = 100

# Test the model on training data
mask = np.random.randint(data['X_train'].shape[0], size=batch_size)
X, y = data['X_train'][mask], data['y_train'][mask]
y_pred = model.loss(X).argmax(axis=1)
print ('Training accuracy: ', (y_pred == y).mean())

# Test the model on validation data
mask = np.random.randint(data['X_val'].shape[0], size=batch_size)
X, y = data['X_val'][mask], data['y_val'][mask]
y_pred = model.loss(X).argmax(axis=1)
print ('Validation accuracy: ', (y_pred == y).mean())

# Saliency Maps
def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.
  
  Input:
  - X: Input images, of shape (N, 3, H, W)
  - y: Labels for X, of shape (N,)
  - model: A PretrainedCNN that will be used to compute the saliency map.
  
  Returns:
  - saliency: An array of shape (N, H, W) giving the saliency maps for the input
    images.
  """
  saliency = None
  ##############################################################################
  # TODO: Implement this function. You should use the forward and backward     #
  # methods of the PretrainedCNN class, and compute gradients with respect to  #
  # the unnormalized class score of the ground-truth classes in y.             #
  ##############################################################################
  N = X.shape[0]
  scores, cache = model.forward(X, mode='test')
  dscores = np.zeros_like(scores)
  dscores[range(N), y] = 1      #ground truth saliency map
  #dscores[range(N), np.argmax(scores, axis=1)] = 1 #top 1 prediction saliency map
  dX, _ = model.backward(dscores, cache)
  saliency = np.max(np.abs(dX), axis=1)
  #pass
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return saliency

def show_saliency_maps(mask):
  mask = np.asarray(mask)
  X = data['X_val'][mask]
  y = data['y_val'][mask]

  saliency = compute_saliency_maps(X, y, model)

  for i in range(mask.size):
    plt.subplot(2, mask.size, i + 1)
    plt.imshow(deprocess_image(X[i], data['mean_image']))
    plt.axis('off')
    plt.title(data['class_names'][y[i]][0])
    plt.subplot(2, mask.size, mask.size + i + 1)
    plt.title(mask[i])
    plt.imshow(saliency[i])
    plt.axis('off')
  plt.gcf().set_size_inches(10, 4)
  plt.show()

# Show some random images
mask = np.random.randint(data['X_val'].shape[0], size=5)
show_saliency_maps(mask)
  
# These are some cherry-picked images that should give good results
show_saliency_maps([128, 3225, 2417, 1640, 4619])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Fooling Images
def make_fooling_image(X, target_y, model): # a method based on softmax loss and regularization
  """
  Generate a fooling image that is close to X, but that the model classifies
  as target_y.
  
  Inputs:
  - X: Input image, of shape (1, 3, 64, 64)
  - target_y: An integer in the range [0, 100)
  - model: A PretrainedCNN
  
  Returns:
  - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
  """
  X_fooling = X.copy()
  ##############################################################################
  # TODO: Generate a fooling image X_fooling that the model will classify as   #
  # the class target_y. Use gradient ascent on the target class score, using   #
  # the model.forward method to compute scores and the model.backward method   #
  # to compute image gradients.                                                #
  #                                                                            #
  # HINT: For most examples, you should be able to generate a fooling image    #
  # in fewer than 100 iterations of gradient ascent.                           #
  ##############################################################################
  N = X.shape[0]
  reg = 5e-5
  from cs231n.layers import softmax_loss
  for i in range(100):
      R = X_fooling - X
      scores, cache = model.forward(X_fooling, mode='test')
      loss, dscores = softmax_loss(scores, target_y)
      loss += 0.5 * reg * np.sum(R * R)
      print ('softmax loss:', loss)
      y_pred = np.argmax(scores)
      print ('target class index', target_y,'current class index:',y_pred)
      if target_y == y_pred:
        print ('iter num:', i)
        break
      else: 
          df, _ = model.backward(dscores, cache)
          #print dX
          dX = reg * R + df
          X_fooling -= 6000*dX
  #passhttp://10.10.7.221:8890/notebooks/assignment3/ImageGradients.ipynb#
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_fooling

def make_fooling_image_navie(X, target_y, model): # a navie method
  """
  Generate a fooling image that is close to X, but that the model classifies
  as target_y.
  
  Inputs:
  - X: Input image, of shape (1, 3, 64, 64)
  - target_y: An integer in the range [0, 100)
  - model: A PretrainedCNN
  
  Returns:
  - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
  """
  X_fooling = X.copy()
  ##############################################################################
  # TODO: Generate a fooling image X_fooling that the model will classify as   #
  # the class target_y. Use gradient ascent on the target class score, using   #
  # the model.forward method to compute scores and the model.backward method   #
  # to compute image gradients.                                                #
  #                                                                            #
  # HINT: For most examples, you should be able to generate a fooling image    #
  # in fewer than 100 iterations of gradient ascent.                           #
  ##############################################################################
  iter = 0
  current_predict = -1
  labels = [target_y]
  while iter<100 and current_predict!= target_y:
     scores, cache = model.forward(X_fooling, mode='test')
     current_predict = scores[0].argmax()  
     if current_predict!= target_y:
          if iter !=0:
            #print scores_old - scores
            print (np.argmax(scores_old - scores))
            labels.append(np.argmax(scores_old - scores).item())
            #print labels
          dscores = np.zeros(scores.shape)
          #dscores[:,target_y] = max(scores[:,current_predict] - scores[:,target_y],100)
          dscores[:,target_y] = 1000
          dscores[:,current_predict] = min(-scores[:,current_predict] + scores[:,target_y],-100)  
          w, _ = model.backward(dscores, cache)
          X_fooling +=w

          iter+=1
          scores_old = scores
          print ("iteration: ", iter, "  predict: ", current_predict)
  #passhttp://10.10.7.221:8890/notebooks/assignment3/ImageGradients.ipynb#
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_fooling,labels


# Find a correctly classified validation image
while True:
  i = np.random.randint(data['X_val'].shape[0])
  X = data['X_val'][i:i+1]
  y = data['y_val'][i:i+1]
  y_pred = model.loss(X)[0].argmax()
  if y_pred == y: break

target_y = 67
#X_fooling, labels = make_fooling_image_navie(X, target_y, model)
X_fooling = make_fooling_image(X, target_y, model)
# Make sure that X_fooling is classified as y_target
scores = model.loss(X_fooling)
assert scores[0].argmax() == target_y, 'The network is not fooled!'

# Show original image, fooling image, and difference
plt.subplot(1, 3, 1)
plt.imshow(deprocess_image(X, data['mean_image']))
plt.axis('off')
plt.title(data['class_names'][int(y)][0])
plt.subplot(1, 3, 2)
plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))
plt.title(data['class_names'][int(target_y)][0])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Difference')
plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))
plt.axis('off')
plt.show()

#def compute_saliency_maps(X, y, model):
#  """
#  Compute a class saliency map using the model for images X and labels y.
#  
#  Input:
#  - X: Input images, of shape (N, 3, H, W)
#  - y: Labels for X, of shape (N,)
#  - model: A PretrainedCNN that will be used to compute the saliency map.
#  
#  Returns:
#  - saliency: An array of shape (N, H, W) giving the saliency maps for the input
#    images.
#  """
#  saliency = None
#  N = X.shape[0]
#  scores, cache = model.forward(X, mode='test')
#  dscores = np.zeros_like(scores)
#  #dscores[range(N), y] = 1      #ground truth saliency map
#  dscores[range(N), np.argmax(scores, axis=1)] = 1 #top 1 prediction saliency map
#  dX, _ = model.backward(dscores, cache)
#  saliency = np.max(np.abs(dX), axis=1)
#
#  return saliency
