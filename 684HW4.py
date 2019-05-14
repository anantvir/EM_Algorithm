import math
import os
import random
import operator

file=open('D:\\Courses\\SPRING19\\MachineLearning\\HW4\\em_data.txt','r') # Initialize path from command line

K=3
# #----------------------------------------------------Code from Hang Chen(taken only for testing purpose, will write own version after testing)---------------------------------------------------------------------
# # use K-means to initialize mu in theta
# data_list = [float(line.rstrip('\n')) for line in file]
# data_length = len(data_list)
# def initialize_mu_K_means():
#     old_centroids = []
#     new_centroids = []
#     # initialize centroids
#     for i in range(K):
#         random_point = data_list[random.randint(0, data_length-1)]
#         while random_point in new_centroids:
#             random_point = data_list[random.randint(0, data_length-1)]
#         new_centroids.append(random_point)
#     # check for convergence
#     while old_centroids != new_centroids:
#         # assign each data point to its closest centroid
#         old_centroids = new_centroids
#         cluster_dict = dict([(key, []) for key in old_centroids])
#         for data_point in data_list:
#             distance_dict = {}
#             for centroid in cluster_dict:
#                 # https://www.geeksforgeeks.org/abs-in-python/
#                 distance_dict[centroid] = abs(data_point - centroid)
#             best_centroid = min(distance_dict.items(), key=operator.itemgetter(1))[0]
#             cluster_dict[best_centroid].append(data_point)
#         # recalculate centroids(mean value, mu) for each cluster
#         new_centroids = []
#         for cluster in cluster_dict:
#             new_centroids.append(sum(cluster_dict[cluster])/len(cluster_dict[cluster]))
#     return new_centroids, cluster_dict
# #-------------------------------------------------------Hangs Code------------------------------------------------------------------

data_list=[]
# Initialize Nested Data Dictionary
for eachLine in file:
    newItem = {eachLine : []}
    data_list.append(newItem)

sumRandNumbers=0.0
randNumbers=[]
for x in range(K):
    randNumbers.append(random.uniform(0,1))
sumRandNumbers = sum(randNumbers)

# Initialize alpha_k.
aplha_k=[]
for x in range(len(randNumbers)):
    aplha_k.append(randNumbers[x]/sumRandNumbers)

sigma=None  
model_params=[[0.2,0.4],[0.22,0.54],[1.23,3.998]]   #After initialization by K-means, we will get model parameters in this form
iterations = 1  # Or get them in command line

n_k_dict=dict()
mu_k_dict=dict()
sigma_k_dict=dict()
for x in range(K):
    n_k_dict[x] = 0
    mu_k_dict[x] = 0
    sigma_k_dict[x] = []

def pmfGaussian(dataPoint,variance,mean):
    return (1/math.sqrt(variance)*math.sqrt(2*math.pi))*math.exp((-(data_instance - mean)**2)/2*variance) 
p=0
#w_ik_x_i = {} # temporary list to store product of w_ik and x_i which is used to calculate mu_k
for x in range(iterations): # Iterations
    print('Running iteration :',p)
    p += 1
    # Expectation(E) Step Begins
    for eachValue in data_list:                             # For each training instance
        data_instance = float(list(eachValue.keys())[0])            
        for eachModel in range(K):                          # For each model
            mean_each_model=model_params[eachModel][0]
            variance_each_model=model_params[eachModel][1]
            sumOfPmfGaussian=0
            for x in range(K):                              #Compute Denominator
                mean = model_params[x][0]
                variance = model_params[x][1]
                aplha_j = aplha_k[x]
                sumOfPmfGaussian += pmfGaussian(data_instance,variance,mean)*aplha_j
            E_w_ik = pmfGaussian(data_instance,variance_each_model,mean_each_model)*aplha_k[eachModel]/sumOfPmfGaussian
            
            list(eachValue.values())[0].append(E_w_ik)      # Store vavalues of E_w_ik in data_list for future calculations

            # M - (Maximization) Step begins, calculating n_k values here (compute all calculations which need iteration over whole training data once i.e inside this for loop for 'i')
            n_k_dict[eachModel] += E_w_ik
            product_w_x = E_w_ik*data_instance
            mu_k_dict[eachModel] += product_w_x

    # -------------------M Step-------------------------------
    for model in range(K): #Since M step calculates model(denoted by k) specific parameters, so loop over k
        aplha_k[model] = n_k_dict[model]/len(data_list)

        mu_k_dict[model] = (1/n_k_dict[model])*mu_k_dict[model]  # Mean of model k (mu_k)

        total=0
        for index in data_list:
            data_instance = float(list(index.keys())[0])
            w_ik = list(index.values())[0][model]
            print(w_ik)
            total += w_ik*(data_instance - mu_k_dict[model])**2
        
        sigma_k_dict[model] = (1/n_k_dict[model])*total     # Variance of model k



print("Specified Iterations Completed")
            



                        

        
        