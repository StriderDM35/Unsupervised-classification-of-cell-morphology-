#https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.io as skio
import umap
import umap.plot
import random
import hdbscan
import seaborn as sns
import pandas as pd
from mpl_toolkits import mplot3d
from selective_search import selective_search as s_search #https://pypi.org/project/selective-search/
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, OPTICS, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import contingency_matrix as cm
from sklearn.metrics import confusion_matrix, v_measure_score, homogeneity_score, completeness_score

from SERNN_1_model import seq_model

#tf.keras.models.Model does not allow for model.save for some strange reasons
import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K


#Load images
def load_image(image_dir):
  image_list = []
  num_img = 0
  for filename in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, filename))
    if img is not None:
      img = np.float16(img)/np.max(img)
      img = np.uint8(img*255)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.float16(img)/np.max(img)
      image_list.append(img)
      num_img += 1
  print("\nImage list shape = ", np.shape(image_list))
  
  image_np = np.empty((num_img, 256, 256, 3), dtype = np.float16)

  for i in range(num_img):
    image_np[i,:,:,:] = image_list[i]

  return image_np


#Create labels
def img_label(img_stack, label_num):
  cell_label = np.empty((len(img_stack)))
  for i in range(len(cell_label)):
    cell_label[i] = label_num

  return cell_label

#Using PCA
def plot_distribution(raw_cell_stack, cell_label_stack, cell_type):
  cell_length = raw_cell_stack.shape[0]
  print("\nCell length = {}".format(cell_length))
  learning_rate = np.uint8(cell_length/12)
  perplexity = 1 #np.uint8(cell_length/100)
  if perplexity <= 0 :
    perplexity = 1
  exagg = 4
  print("\nTSNE parameters used: LR = {}, P = {}, Ex = {}".format(learning_rate, perplexity, exagg))
  reduced_cell_stack = TSNE(n_components = 2, perplexity = 4, early_exaggeration = 4, 
    learning_rate = learning_rate, random_state = 42, init = "pca").fit_transform(raw_cell_stack)

  cell_list = ["MDCK C1", "MDCK C4", "MDCK FN", "MDCK LN", "MDCK LN10", "MDCK VN", "MEF C1", 'MEF C4', "MEF FN", "MEF LN", "MEF LN10", "MEF VN"]

  #T-SNE distribution
  reduced_x = []
  reduced_y = []
  for j in range(len(reduced_cell_stack)):
    reduced_x.append(reduced_cell_stack[j][0])
    reduced_y.append(reduced_cell_stack[j][1])
  fig, ax = plt.subplots()
  a0 = plt.scatter(reduced_x[:30], reduced_y[:30], c = 'r', marker = 'p', alpha = 0.5)
  a1 = plt.scatter(reduced_x[30:60], reduced_y[30:60], c = 'g', marker = '*', alpha = 0.5)
  a2 = plt.scatter(reduced_x[60:90], reduced_y[60:90], c = 'b', marker = 'v', alpha = 0.5)
  a3 = plt.scatter(reduced_x[90:120], reduced_y[90:120], c = 'c', marker = '^', alpha = 0.5)
  a4 = plt.scatter(reduced_x[120:150], reduced_y[120:150], c = 'peru', marker = '<', alpha = 0.5)
  a5 = plt.scatter(reduced_x[150:180], reduced_y[150:180], c = 'y', marker = '>', alpha = 0.5)

  a6 = plt.scatter(reduced_x[180:210], reduced_y[180:210], c = 'y', marker = 'p', alpha = 0.5)
  a7 = plt.scatter(reduced_x[210:240], reduced_y[210:240], c = 'peru', marker = '*', alpha = 0.5)
  a8 = plt.scatter(reduced_x[240:270], reduced_y[240:270], c = 'c', marker = 'v', alpha = 0.5)
  a9 = plt.scatter(reduced_x[270:300], reduced_y[270:300], c = 'b', marker = '^', alpha = 0.5)
  a10 = plt.scatter(reduced_x[300:330], reduced_y[300:330], c = 'g', marker = '<', alpha = 0.5)
  a11 = plt.scatter(reduced_x[330:360], reduced_y[330:360], c = 'r', marker = '>', alpha = 0.5)

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend = ax.legend([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11], cell_list, loc = "center left", title = "Classes", bbox_to_anchor = (1, 0.5))
  #plt.legend([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], cell_list, loc = 'best')
  ax.add_artist(legend)
  title = cell_type + " " + "t-SNE distribution"
  plt.suptitle(title)
  cell_name = cell_type + "_"
  name = path + "/TSNE_" + cell_name + "plot.png"
  plt.savefig(name)
  plt.close()
  print("\nDistribution mapping saved...")

  #PCA distribution
  pca = PCA(n_components = 2)
  reduced_cell_stack_pca = pca.fit_transform(raw_cell_stack)
  reduced_x = []
  reduced_y = []
  for j in range(len(reduced_cell_stack_pca)):
    reduced_x.append(reduced_cell_stack_pca[j][0])
    reduced_y.append(reduced_cell_stack_pca[j][1])
  fig, ax = plt.subplots()
  a0 = plt.scatter(reduced_x[:30], reduced_y[:30], c = 'r', marker = 'p', alpha = 0.5)
  a1 = plt.scatter(reduced_x[30:60], reduced_y[30:60], c = 'g', marker = '*', alpha = 0.5)
  a2 = plt.scatter(reduced_x[60:90], reduced_y[60:90], c = 'b', marker = 'v', alpha = 0.5)
  a3 = plt.scatter(reduced_x[90:120], reduced_y[90:120], c = 'c', marker = '^', alpha = 0.5)
  a4 = plt.scatter(reduced_x[120:150], reduced_y[120:150], c = 'peru', marker = '<', alpha = 0.5)
  a5 = plt.scatter(reduced_x[150:180], reduced_y[150:180], c = 'y', marker = '>', alpha = 0.5)

  a6 = plt.scatter(reduced_x[180:210], reduced_y[180:210], c = 'y', marker = 'p', alpha = 0.5)
  a7 = plt.scatter(reduced_x[210:240], reduced_y[210:240], c = 'peru', marker = '*', alpha = 0.5)
  a8 = plt.scatter(reduced_x[240:270], reduced_y[240:270], c = 'c', marker = 'v', alpha = 0.5)
  a9 = plt.scatter(reduced_x[270:300], reduced_y[270:300], c = 'b', marker = '^', alpha = 0.5)
  a10 = plt.scatter(reduced_x[300:330], reduced_y[300:330], c = 'g', marker = '<', alpha = 0.5)
  a11 = plt.scatter(reduced_x[330:360], reduced_y[330:360], c = 'r', marker = '>', alpha = 0.5)

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend = ax.legend([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11], cell_list, loc = "center left", title = "Classes", bbox_to_anchor = (1, 0.5))
  #plt.legend([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], cell_list, loc = 'best')
  ax.add_artist(legend)
  cell_name = cell_type + "_"
  title = cell_type + " " + "PCA distribution"
  plt.suptitle(title)
  name = path + "/PCA_" + cell_name + "plot.png"
  plt.savefig(name)
  plt.close()
  print("\nDistribution mapping saved...")

  return reduced_cell_stack

#Perform clustering
def hdbscan_cluster(reduced_cell_stack, cell_type):
  hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = 5)
  cluster_result = hdbscan_model.fit_predict(reduced_cell_stack)
  reduced_x = []
  reduced_y = []
  for j in range(len(reduced_cell_stack)):
    reduced_x.append(reduced_cell_stack[j][0])
    reduced_y.append(reduced_cell_stack[j][1])
  fig, ax = plt.subplots()
  scatter = ax.scatter(reduced_x, reduced_y, c = cluster_result, alpha = 0.75)
  for i, text in enumerate(cluster_result):
    ax.annotate(text, (reduced_x[i], reduced_y[i]))
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  legend = ax.legend(*scatter.legend_elements(), loc = "center left", title = "Classes", bbox_to_anchor = (1, 0.5))
  ax.add_artist(legend)
  title = cell_type + " " + "clustering result"
  plt.suptitle(title)
  name = path + "/HDBSCAN_" + cell_type + "_result" + ".png"
  plt.savefig(name)
  plt.close()

  #Build the cluster hierarchy (dendrogram)
  hdbscan_model.single_linkage_tree_.plot(cmap = "viridis", colorbar = True)
  name = path + "/Dendrogram_" + cell_type + "_result" + ".png"
  plt.savefig(name)
  plt.close()

  #Condensed cluster tree
  hdbscan_model.condensed_tree_.plot(select_clusters = True, selection_palette = sns.color_palette(), label_clusters = True)
  name = path + "/Condensed_labelled_" + cell_type + "_result" + ".png"
  plt.savefig(name)
  plt.close()

  hdbscan_model.condensed_tree_.plot(select_clusters = True, selection_palette = sns.color_palette())
  name = path + "/Condensed_" + cell_type + "_result" + ".png"
  plt.savefig(name)
  plt.close()

  return cluster_result

#Define functions for plotting contingency matrix
def plot_contingency_matrix(cluster_result, path, cell_label_stack):
    """If you prefer color and a colorbar"""
    matrix = cm(cell_label_stack, cluster_result)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.yticks(np.arange(12), ("MDCK C1", "MDCK C4", "MDCK FN", "MDCK_LN", "MDCK LN10", "MDCK VN", 
      "MEF_C1", "MEF C4", "MEF FN", "MEF_LN", "MEF LN10", "MEF VN"))
    #plt.yticks(np.arange(5), ("MEF C4", "MEF FN", "MEF LN10", "MEF VN"))
    fig.colorbar(cax)
    cax.set_clim(0, 8)
    name = path + "MDCK_MEF_contingency_matrix.png"
    plt.savefig(name)

def plot_confusion_matrix(predictions, ground_truth, path):
  matrix = confusion_matrix(ground_truth, predictions)
  annotations = np.empty_like(matrix).astype(str)
  print("Confusion matrix = ", matrix)
  fig = plt.figure(figsize = (10,10))
  ax = fig.add_subplot(111)
  cax = ax.matshow(matrix)
  plt.yticks(np.arange(12), ("MDCK C1", "MDCK C4", "MDCK FN", "MDCK_LN", "MDCK LN10", "MDCK VN", 
    "MEF_C1", "MEF C4", "MEF FN", "MEF_LN", "MEF LN10", "MEF VN"))
  #plt.yticks(np.arange(5), ("MEF C4", "MEF FN", "MEF LN10", "MEF VN"))
  fig.colorbar(cax)
  cax.set_clim(0, 20)
  name = path + "MDCK_MEF_confusion_matrix.png"
  plt.savefig(name)
  plt.close()

  #Plot a heatmap showing the percentages
  matrix_sum = np.sum(matrix, axis = 1, keepdims = True)
  matrix_perc = (matrix / 30)*100.0
  nrows, ncols = matrix.shape
  for i in range(nrows):
    for j in range(ncols):
      c = matrix[i,j]
      p = matrix_perc[i,j]
      if i == j:
        s = matrix_sum[i]
        annotations[i,j] = '%.1f%%\n%d/30' % (p, c)
      elif c == 0:
        annotations[i,j] = ''
      else:
        annotations[i,j] = '\n%d' % (c)
  labels = ["MDCK C1", "MDCK C4", "MDCK FN", "MDCK LN", "MDCK LN10", "MDCK VN", "MEF C1", "MEF C4", "MEF FN", "MEF LN", "MEF LN10", "MEF VN"]
  matrix_pd = pd.DataFrame(matrix, index = labels, columns = labels)
  matrix_pd.index.name = "Ground Truth"
  matrix_pd.columns.name = "Prediction"
  fig, ax = plt.subplots(figsize = (10,10))
  sns.heatmap(matrix_pd, annot = annotations, fmt = '', cmap = "YlGnBu", square = True)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  name = path + "MDCK_MEF_heatmap_2.png"
  plt.savefig(name)
  plt.close()

#Save the images and check
def generate_labelled_images(img_stack, cluster_label_stack, classify_label_stack, ground_truth_stack, path_origin):
  cluster_path = path_origin + "cluster/"
  try: 
    os.listdir(cluster_path)
  except FileNotFoundError:
    os.mkdir(cluster_path)
  else:
    print("Directory found")

  classify_path = path_origin + "classification/"
  try: 
    os.listdir(classify_path)
  except FileNotFoundError:
    os.mkdir(classify_path)
  else:
    print("Directory found")

  outlier_folder = "outlier/"
  outlier_path = cluster_path + outlier_folder
  try: 
    os.listdir(outlier_path)
  except FileNotFoundError:
    os.mkdir(outlier_path)
  else:
    print("Directory found")

  for i in range(len(img_stack)):
    img = img_stack[i]

    img = np.float32(img)/np.max(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Convert output image to type uint8
    output_img = np.float32(img)/np.max(img)
    output_img = np.uint8(output_img*255)

    cluster_label = cluster_label_stack[i]
    class_label = classify_label_stack[i]
    ground_truth = ground_truth_stack[i]

    if cluster_label < 0:
      cluster_name = outlier_path + "/" + "(" + str(cluster_label) + ")_" + str(i) + '.tif'
    else:
      cluster_name = cluster_path + "/" + "(" + str(cluster_label) + ")_" + str(i) + '.tif'

    classify_name = classify_path + "/" + "(" + str(class_label) + ")_" + str(ground_truth) + "_" + str(i) + '.tif'

    skio.imsave(cluster_name, output_img)
    skio.imsave(classify_name, output_img)


def evaluate_clustering(cell_label, cell_cluster):
  ground_truth = []
  cluster_pred = []
  #In order to use v-measure properly, we need to remove the outliers first
  for i in range(len(cell_label)):
    if cell_cluster[i] >= 0:
      ground_truth.append(cell_label[i])
      cluster_pred.append(cell_cluster[i])
  v_score = v_measure_score(ground_truth, cluster_pred) #leave beta = 1
  print("\nv-measure score = %.6f" % v_score)

  homo_score = homogeneity_score(ground_truth, cluster_pred)
  com_score = completeness_score(ground_truth, cluster_pred)

  print("\nHomogeneity score = {} || Completenss score = {}".format(homo_score, com_score))

def grad_cam_old(path, cell_stack_4D, cell_label_stack, model, predicted_class):
#https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
  path_to_save = path + "grad_cam_analysis/"
  try: 
    os.listdir(path_to_save)
  except FileNotFoundError:
    os.mkdir(path_to_save)
  else:
    print("Directory found")


  print("\nPerforming Grad-CAM analysis now...")

  for i in range(len(cell_stack_4D)):
    test_img = np.empty((1, 256, 256, 3))
    test_img[0] = cell_stack_4D[i]

    grad_model = Model([model.inputs], [model.get_layer("elu_35").output, model.output])

    with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(test_img)
      tape.watch(conv_outputs)
      pred_index = tf.argmax(predictions[0])
      class_index = int(cell_label_stack[i]) #It must NOT have a numpy wrapper
      print("\nClass index: ", class_index)
      top_class_channel = predictions[:, pred_index]
      print("i: ", str(i))

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    output_np = conv_outputs.numpy()[0]
    pooled_grads_np = pooled_grads.numpy()

    for j in range(pooled_grads_np.shape[-1]):
      output_np[:,:,j] *= pooled_grads_np[j]

    cam = np.mean(output_np, axis = -1) #Average over all the arrays to get a single 2D array
    cam = np.clip(cam, 0, np.max(cam)) / np.max(cam)
    cam = np.uint8(cam*255)
    cam = cv2.resize(cam, (256,256))
    name = path_to_save + "/cam/img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) + "_" + str(predicted_class[i]) + ".tif"
    skio.imsave(name, cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_VIRIDIS)
    img = test_img[0]
    img = np.float32(img)/np.max(img)
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    name = path_to_save + "img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) + "_" + str(predicted_class[i]) + ".tif"
    skio.imsave(name, output)

    cam_name = path_to_save + "img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) + "_" + str(predicted_class[i]) +  "_grad.tif"
    skio.imsave(cam_name, heatmap)

def grad_cam(path, cell_stack_4D, cell_label_stack, model):
#https://gist.github.com/RaphaelMeudec/e9a805fa82880876f8d89766f0690b54
  path_to_save = path + "gradcam_cluster/"
  try: 
    os.listdir(path_to_save)
  except FileNotFoundError:
    os.mkdir(path_to_save)
  else:
    print("Directory found")


  print("\nPerforming Grad-CAM analysis now...")

  for i in range(len(cell_stack_4D)):
    test_img = np.empty((1, 256, 256, 3))
    test_img[0] = cell_stack_4D[i]

    grad_model = Model([model.inputs], [model.layers[-4].output, model.output])

    with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(test_img)
      tape.watch(conv_outputs)
      pred_index = tf.argmax(predictions[0])
      #class_index = int(cell_label_stack[i]) #It must NOT have a numpy wrapper
      #print("\nClass index: ", class_index)
      top_class_channel = predictions[:, pred_index]
      #print("i: ", str(i))

    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    output_np = conv_outputs.numpy()[0]
    pooled_grads_np = pooled_grads.numpy()

    for j in range(pooled_grads_np.shape[-1]):
      output_np[:,:,j] *= pooled_grads_np[j]

    cam = np.mean(output_np, axis = -1) #Average over all the arrays to get a single 2D array
    cam = np.clip(cam, 0, np.max(cam)) / np.max(cam)
    cam = np.uint8(cam*255)
    cam = cv2.resize(cam, (256,256))
    #name = path_to_save + "/cam/img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) + ".tif"
    #skio.imsave(name, cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_VIRIDIS)
    img = test_img[0]
    img = np.float32(img)/np.max(img)
    img = np.uint8(img*255)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    name = path_to_save + "img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) + ".tif"
    skio.imsave(name, output)

    cam_name = path_to_save + "img_" + str(np.uint8(cell_label_stack[i])) + "_" + str(i) +  "_grad.tif"
    skio.imsave(cam_name, heatmap)



#RUN SCRIPT BELOW

#Load images 
print("\nLoading images...")
cell_1_read = load_image("full_data/MDCK_C1/")
cell_2_read = load_image("full_data/MDCK_C4/")
cell_3_read = load_image("full_data/MDCK_FN/")
cell_4_read = load_image("full_data/MDCK_LN/")
cell_5_read = load_image("full_data/MDCK_LN10/")
cell_6_read = load_image("full_data/MDCK_VN/")
cell_7_read = load_image("full_data/MEF_C1/")
cell_8_read = load_image("full_data/MEF_C4/")
cell_9_read = load_image("full_data/MEF_FN/")
cell_10_read = load_image("full_data/MEF_LN/")
cell_11_read = load_image("full_data/MEF_LN10/")
cell_12_read = load_image("full_data/MEF_VN/")

cell_1 = cell_1_read[50:]
cell_2 = cell_2_read[50:]
cell_3 = cell_3_read[50:]
cell_4 = cell_4_read[50:]
cell_5 = cell_5_read[50:]
cell_6 = cell_6_read[50:]
cell_7 = cell_7_read[50:]
cell_8 = cell_8_read[50:]
cell_9 = cell_9_read[50:]
cell_10 = cell_10_read[50:]
cell_11 = cell_11_read[50:]
cell_12 = cell_12_read[50:]


cell_stack = np.vstack((cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7, cell_8, cell_9, cell_10, cell_11, cell_12))
#cell_stack = np.vstack((cell_1, cell_2, cell_3))
print("\nShape of cell stack: ", np.shape(cell_stack))

cell_stack_flatten = cell_stack.reshape(-1, 65536)
cell_stack_4D = cell_stack.reshape(-1, 256, 256, 3)

#Load model
filename = '11-JAN-2022_SERNN_1_model.h5'
#filename = '31-MAY-2021_SERNN_1_model.h5'
model = load_model(filename)
layer_outputs = [layer.output for layer in model.layers[:]]
ext_model = Model(inputs = model.input, outputs = layer_outputs)

model_class = np.empty((len(cell_stack_4D)))
model_cluster = np.empty((len(cell_stack_4D), 128))

print("\nExtracting classification and clustering results now...")
for i in range(len(cell_stack_4D)):
  img = cell_stack_4D[i]
  img_4D = img.reshape(-1, 256, 256, 3)
  model_results_all_layers = ext_model.predict(img_4D)
  #if i == 0:
  #  print("\nLayers available:")
  #  for i in range(len(model_results_all_layers)):
  #    print(np.shape(model_results_all_layers[i]))

  model_class[i] = np.uint8(np.argmax(model_results_all_layers[-1]))
  model_cluster[i] = model_results_all_layers[-3]

print(np.shape(model_cluster))

#plt.imshow(model_cluster, cmap = "hot", aspect = 0.2)
#plt.show()


#Create directory if it hasn't been created yet
path = "SERNN_result/result_16v5_SERNN_(interpretation)/"
try: 
  os.listdir(path)
except FileNotFoundError:
  os.mkdir(path)
else:
  print("Directory found")

cell_1_label = img_label(cell_1, 0)
cell_2_label = img_label(cell_2, 1)
cell_3_label = img_label(cell_3, 2)
cell_4_label = img_label(cell_4, 3)
cell_5_label = img_label(cell_5, 4)  
cell_6_label = img_label(cell_6, 5)

cell_7_label = img_label(cell_7, 6)
cell_8_label = img_label(cell_8, 7)
cell_9_label = img_label(cell_9, 8)
cell_10_label = img_label(cell_10, 9)
cell_11_label = img_label(cell_11, 10)
cell_12_label = img_label(cell_12, 11)

cell_label_stack = np.hstack((cell_1_label, cell_2_label, cell_3_label, cell_4_label, cell_5_label, cell_6_label, 
  cell_7_label, cell_8_label, cell_9_label, cell_10_label, cell_11_label, cell_12_label))
#cell_label_stack = np.hstack((cell_6_label, cell_7_label, cell_8_label, cell_9_label))

cell_distribution = plot_distribution(model_cluster, cell_label_stack, "MDCK_MEF")
cell_cluster = hdbscan_cluster(cell_distribution, "MDCK_MEF")
#plot_confusion_matrix(cell_label_stack, model_class, path)

print(cell_cluster)

#We extract the clusters that we want and we perform GradCAM analysis on them
cluster_list = []
cell_list = []
for i in range(len(cell_cluster)):
  cluster = cell_cluster[i]
  if cluster == 0 or cluster == 12:
    cluster_list.append(cluster)
    cell_list.append(cell_stack_4D[i])

print(cluster_list)
print(np.shape(cell_list))

#We perform dimming of the channels where required

grad_cam(path, cell_list, cluster_list, model)

print(STOP)

#We blank out the channels here
#R - actin, G - microtubule, B - nucleus
#cell_one_channel = cell_stack.copy() 
#cell_one_channel[:,:,:,0] = 0
#cell_one_channel[:,:,:,2] = 0
#model_cluster = np.empty((len(cell_stack_4D), 128))
#for i in range(len(cell_stack_4D)):
#  img = cell_one_channel[i]
#  img_4D = img.reshape(-1, 256, 256, 3)
#  model_results_all_layers = ext_model.predict(img_4D)
#  model_cluster[i] = model_results_all_layers[-3]

#labelled_map = np.empty((len(cell_stack_4D)+130, 128))
#labelled_map = []
#sorted_list = []
#for i in range(len(cell_stack_4D)):
#  instance = []
#  maps = np.asarray(model_cluster[i,:]) #This is an array
#  label = cell_cluster[i]
  #if label < 0 :
  #  label = 100 #Avoid negative labels
#  instance.append(label)
#  instance.append(maps)
#  sorted_list.append(instance)

#sorted_map = sorted(sorted_list, key = lambda x: x[0]) #Only sort based on the first element of the tuples
#blank_map = np.zeros((10, 128)) 

#j = 0
#for i in range(len(cell_stack_4D)+130):
  #label = sorted_map[i][0]
  #labelled_map[j] = sorted_map[i][1]
  #try:
    #next_label = sorted_map[i+1][0]
  #except IndexError: 
    #print("We have reached the last index")
    #break
  #if next_label != label:
    #labelled_map = np.append(labelled_map, blank_map, axis = 0)
    #labelled_map[j+1:j+11] = blank_map
    #j += 10
    #labelled_map.append(blank_map)

#  j += 1

#fig, ax = plt.subplots()
#ax.imshow(labelled_map, cmap = 'Greens', aspect = 0.2)
#ax.set_yticklabels([]) #turn off tick labels
#ax.set_yticks([]) #turn off ticks
#plt.show()

#plot_contingency_matrix(cell_cluster, path, cell_label_stack)
#plot_confusion_matrix(cell_label_stack, model_class, path)
#generate_labelled_images(cell_stack, cell_cluster, model_class, cell_label_stack, path)
#evaluate_clustering(cell_label_stack, cell_cluster)
#grad_cam(path, cell_stack_4D, cell_label_stack, model, model_class)



