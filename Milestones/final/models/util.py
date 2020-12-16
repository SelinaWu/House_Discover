# import libraries
from __future__ import absolute_import
from __future__ import print_function

import shap
import pickle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from numpy import dot
from numpy.linalg import norm
from itertools import repeat


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, Layer, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Model


from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate

from sklearn.metrics.pairwise import cosine_similarity

############
### Data ###
############
def load_dataset(data_src="window_full"):
    '''
      Load in saved anchor, positive, negative dataset.
      `data_src`: the type of data of either "window_full" or "w_image". 
            "w_image" contains 7 house image style features,
            "window_full" contains 77 house info
            Default value "window_full".
    '''
    anchor = np.load(f'./data/anchor_data_{data_src}.npy',allow_pickle=True)
    positive = np.load(f'./data/pos_data_{data_src}.npy',allow_pickle=True)
    negative = np.load(f'./data/neg_data_{data_src}.npy',allow_pickle=True)
    ## generate fake y (not used for training)
    y_train = np.random.randint(0,1, (anchor.shape[0],1))

    print("Loading data:\nAnchor: ", anchor.shape)
    print("Positive: ", positive.shape)
    print("Negative: ", negative.shape)

    return anchor, positive, negative, y_train

def train_test_split(anchor, positive, negative, y_train, train_pct=0.9):
    '''
      run train test split by given train percentage
    '''
    np.random.seed(297)
    train_idx = np.random.choice(np.arange(anchor.shape[0]),size=int(anchor.shape[0]*train_pct),replace=False)
    anchor_train = anchor[train_idx,:]
    pos_train = positive[train_idx,:]
    neg_train = negative[train_idx,:]
    y_training = y_train[train_idx,:]

    anchor_test = np.delete(anchor,train_idx,axis=0)
    pos_test = np.delete(positive,train_idx,axis=0)
    neg_test = np.delete(negative,train_idx,axis=0)
    y_test = np.delete(y_train,train_idx)

    print(anchor.shape)
    print(anchor_train.shape)
    print(anchor_test.shape)

    return anchor_train, pos_train, neg_train, y_training, anchor_test, pos_test, neg_test, y_test


def prepare_contrastive_pair(anchor, pos, neg):
    ## form pairs for contrastive loss model 
    pair_input1 = np.concatenate((anchor,anchor))
    pair_input2 = np.concatenate((pos, neg))

    class0=np.zeros((anchor.shape[0],1))
    class1=np.ones((anchor.shape[0],1))
    output = np.concatenate((class1, class0))
    return pair_input1, pair_input2, output

#############
### Model ###
#############
def create_embedding_model(input_shape):
    '''
    	embedding model for triplet loss
    	Note: do not use dropout layer, it will not increase the model performance
    		  additional dense layer will not increase the model performance
    '''
    input_listing = Input(shape=input_shape)
    x = Dense(32, activation='relu', input_shape=input_shape, name='dense1')(input_listing)
    x = BatchNormalization()(x)
    bottleneck = Dense(16, activation='linear', input_shape=input_shape, name='dense4')(x)
    
    triplet_model_embedding = Model(input_listing, bottleneck)  
    return triplet_model_embedding

## Siamese in triplet loss 
def triplet_loss(x, ALPHA=0.75):

    anchor, positive, negative = x
    
    #Modifying the triplet loss
    anchor = 2*anchor

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    loss = tf.divide(tf.maximum(loss, 0.0), 2.0) #??

    return loss

def build_triplet_model(input_shape, triplet_model_embedding):
    '''
      build siamese model with triplet loss function 
      Note: additional dense layer will not increase the model performance
    '''
    anchor_example = Input(shape=input_shape, name='anchor')
    positive_example = Input(shape=input_shape, name='positive')
    negative_example = Input(shape=input_shape, name='negative')

    positive_embedding = triplet_model_embedding(positive_example)       
    negative_embedding = triplet_model_embedding(negative_example)       
    anchor_embedding = triplet_model_embedding(anchor_example)           

    #The Triplet Model which optimizes over the triplet loss.       
    loss = Lambda(triplet_loss, output_shape=(1,))([anchor_embedding, positive_embedding, negative_embedding])
    triplet_model = Model(inputs=[anchor_example, positive_example, negative_example], 
                          outputs=loss)
    adam_opt = tf.optimizers.Adam(lr=0.001)           
                                                                    
    display(triplet_model.summary())

    triplet_model.compile(loss='mean_absolute_error', optimizer=adam_opt)
    return triplet_model

## contrastive loss
""" Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
"""

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def build_contrastive_loss_model(input_shape, contrastive_model_embedding):
    # Create the 2 inputs
    input1 = Input(shape=input_shape, name='target')
    input2 = Input(shape=input_shape, name='candidate')

    # Share base network with the 2 inputs
    input1_embedding = contrastive_model_embedding(input1)       
    input2_embedding = contrastive_model_embedding(input2)       

    L1_layer = Lambda(cosine_distance)
    L1_distance = L1_layer([input1_embedding, input2_embedding])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    # Define the trainable model
    model = Model(inputs=[input1, input2], 
                  outputs=prediction)

    model.compile(optimizer=Adam(0.001),
                  loss=contrastive_loss)#
    print(model.summary())
    return model

########################
### Model Evaluation ###
########################
def evaluate_model(model,test_data, training_results):
    
    # Get the model train history
    model_train_history = training_results.history
    # Get the number of epochs the training was run for
    num_epochs = len(model_train_history["loss"])

    # Plot training results
    fig, ax = plt.subplots(figsize=(15,5))
    plt.title('Loss')
    # Plot all metrics
    for metric in ["loss","val_loss"]:
        plt.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
    plt.legend()
    plt.show()
    
    # Evaluate on test data
    evaluation_results = model.evaluate(test_data)
    print("Evaluation Results:", evaluation_results)

def eval_pos_is_high(model_embedding, anchor_test, pos_test, neg_test):
    house1 = model_embedding.predict(anchor_test)
    house2 = model_embedding.predict(pos_test)
    house3 = model_embedding.predict(neg_test)

    cos_sim_all = []
    for h1, h2,h3 in zip(house1, house2, house3):
        cos_sim_pos = dot(h1, h2)/(norm(h1)*norm(h2))
        cos_sim_neg = dot(h1, h3)/(norm(h1)*norm(h3))
        cos_sim_all.append(cos_sim_pos-cos_sim_neg)

    cos_sim_all = np.array(cos_sim_all)
    pos_is_high_sum = sum(cos_sim_all>0)
    score = round(pos_is_high_sum/len(anchor_test), 4)*100
    print(f"Positive is higher than negative: {score} %")

def get_shap_interpretation(model_nm, input_data, pick_k=100, max_display=20):
    # select a set of background examples to take an expectation over
    rand_idx = np.random.choice(input_data.shape[0], pick_k, replace=False)
    background = input_data[rand_idx]

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model_nm, background)
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap_values = e.explainer.shap_values(input_data[1:100])
    # plot the feature attributions
    # max_display = 20 # only bar works
    shap.summary_plot(shap_values, input_data[1:100], 
                      plot_type="bar", max_display=max_display) 

##################
### Evaluation ### precision @ k, recall @ k
##################
def calculate_embedding_score(feature_dim, triplet_model_embedding, 
                              data_src="w_image", row_size=15000,
                              is_cos=False):
    cos_sim_all = np.array([])
    ips = np.array([])
    rex_urls = np.array([])

    data_src = "_"+data_src+"s" if data_src=="w_image" else data_src
    df_chunks = pd.read_csv(f"./data/user_medianed_paired_unseen{data_src}.csv",sep=',', 
                            chunksize=row_size)

    print("Load in unseen pairs chunk by chunk: ")
    for eval_df in tqdm(df_chunks):
        ips = np.concatenate([ips, eval_df.ip])
        rex_urls = np.concatenate([rex_urls, eval_df.rex_url])
        wanted_cols = eval_df.columns.difference(['Unnamed: 0','ip','rex_url'], sort=False)
    
        if is_cos:
            anchor_embd = eval_df[wanted_cols].iloc[:, :feature_dim].values
            house_candidate_embd = eval_df[wanted_cols].iloc[:, feature_dim:].values
        else:
            anchor_embd = triplet_model_embedding.predict(eval_df[wanted_cols].iloc[:, :feature_dim].values)
            house_candidate_embd = triplet_model_embedding.predict(eval_df[wanted_cols].iloc[:, feature_dim:].values)
    
        cos_sim_scores = cosine_similarity(anchor_embd, house_candidate_embd)
        cos_sim_scores = np.diag(cos_sim_scores)
        cos_sim_all = np.concatenate([cos_sim_all, cos_sim_scores.flatten()])


    cos_sim_all = np.array(cos_sim_all)
    temporal_pred = np.ones(cos_sim_all.shape)

    # Create prediction result table
    all_pairs_predictions = pd.DataFrame({'ip':ips,'rex_url':rex_urls,'proba':cos_sim_all,'pred':temporal_pred})
    all_pairs_predictions.sort_values('proba',ascending=False,inplace=True)
    all_pairs_predictions.reset_index(drop=True,inplace=True)

    return all_pairs_predictions
	
def precision_recall_at_k(all_pairs_predictions, user_listing_test, k_in):
    # get top k predictions for each ip
    g = all_pairs_predictions.groupby(['ip']).apply(lambda x: x.sort_values(['proba'], ascending = False)).reset_index(drop=True)
    top_k_each = g.groupby('ip').head(k_in)
    recommended_at_k = top_k_each[top_k_each.pred==1].groupby('ip')['rex_url'].apply(list).reset_index()

    recall_results = {}
    precision_results = {}
    #loop through each ip and compare recommended to real test set
    for idx, row in recommended_at_k.iterrows():
      ip_curr = row['ip']
      recommended_at_k_curr = row['rex_url']

      true_for_ip_curr = user_listing_test[user_listing_test.ip==ip_curr].rexUrl.values

      # Recommendations that are relevant
      # find overlap
      correctly_found_ip_curr = list(set(recommended_at_k_curr) & set(true_for_ip_curr))

      recall_results[ip_curr] = len(correctly_found_ip_curr)/len(true_for_ip_curr)
      precision_results[ip_curr] = len(correctly_found_ip_curr)/len(recommended_at_k_curr)


    return recommended_at_k, recall_results, precision_results