# -*- coding: utf-8 -*-
"""
Created on Mon Sep 9 2019

@author: HML TEAM
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        

k = 10

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
#model.predict(11979 , 7729)

similarity_matrix  = ml.computeSimilarityMatrix(data.build_full_trainset().n_items, trainSet)
print('similarity_matrix'  ,similarity_matrix.shape)
simsMatrix = similarity_matrix
#model.compute_similarities()

def getPredictionsForUser(user_id):
    print('Get Predictions...')
    print('simsMatrix' ,simsMatrix.shape)
    testUserInnerID = trainSet.to_inner_uid(user_id)
    #print('testUserInnerID'  ,testUserInnerID)
    #real_user_id = trainSet.to_raw_uid(72)
    #print('real_user_id' ,real_user_id)
    # Get the top K items we rated
    testUserRatings = trainSet.ur[testUserInnerID]
    print('testUserRatings' , testUserRatings)
    kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])
    
    print('kNeighbors' ,kNeighbors)
    # Get similar items to stuff we liked (weighted by rating)
    candidates = defaultdict(float)
    for itemID, rating in kNeighbors:
        print('...')
        similarityRow = simsMatrix[itemID]
        for innerID, score in enumerate(similarityRow):
            candidates[innerID] += score * (rating / 44.0)
    
    # Build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in trainSet.ur[testUserInnerID]:
        watched[itemID] = 1
        
    # Get top-rated items from similar users:
    pos = 0
    
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            print(ml.getMovieName(int(movieID)), ratingSum)
            pos += 1
            if (pos > 50):
                break

    

getPredictionsForUser('7441')

