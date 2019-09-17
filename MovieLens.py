import os
import csv
import sys
import re
import math
from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class MovieLens:

    movieID_to_name = {}
    name_to_movieID = {}
    '''
    ratingsPath = '../ml-latest-small/ratings.csv'
    moviesPath = '../ml-latest-small/movies.csv'
    
    ratingsPath = '../ml-latest-small/product_views_no_zero.csv'
    moviesPath = '../ml-latest-small/products_new.csv
    moviesPath = '../ml-latest-small/products_with_subcategory.csv'
    '''
    
    ratingsPath = '../ml-latest-small/product_views_no_zero.csv'
   
    productPath = '../ml-latest-small/product_dataset.csv'
    
    def loadMovieLensLatestSmall(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        with open(self.productPath, newline='', encoding='UTF-8') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID

        return ratingsDataset
    


    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings
    
    def getProductsData(self):
        print('Loading Product Data...')
        data = defaultdict(list)
        with open(self.productPath, newline='', encoding='UTF-8') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                product_id = int(row[0])
                supplier_id = int(row[2])
                category_id = int(row[3])
                subcategory_id = int(row[4])
                data[product_id] =[supplier_id,category_id ,subcategory_id]
        print('Product Data Loaded Successfully...')
        return data
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesPath, newline='', encoding='UTF-8') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='UTF-8') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                movieID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def getMovieID(self, movieName):
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0
        
        
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        #print('genres1=' , genres1 , 'genres2' , genres2)
        #print('movie1=' , movie1 , 'movie2' , movie2)
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        #print("Before_returning",sumxx,sumyy , sumxy  )
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeProductsSimilarity(self, product1, product2, data):
        score = 0
        #Supplier
        supplier1 = data[product1][0]
        supplier2 = data[product2][0]
        #Category
        category1 = data[product1][1]
        category2 = data[product2][1]
        #Subcategories
        subcategory1 = data[product1][2]
        subcategory2 = data[product2][2]
        
        if(supplier1 == supplier2):
            score += 0.85
        if(category1 == category2):
            score += 0.5
        if(subcategory1 == subcategory2):
            score += 0.6
        #print('score=' , score )
        
        return score
        
        
        
        
    
    def computeSimilarityMatrix(self,trainset_n_items,trainset):
        print("Computing content-based similarity matrix...")
        #genres = self.getGenres()
        # Compute genre distance for every movie combination as a 2x2 matrix
        data = self.getProductsData()
        similarities = np.zeros((trainset_n_items, trainset_n_items))
        print("TRAINSET101", trainset )
        for thisRating in range(trainset_n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", trainset_n_items)
            for otherRating in range(thisRating+1, trainset_n_items):
                #print("thisRating", thisRating, 'otherRating' , otherRating , trainset_n_items)
                thisMovieID = int(trainset.to_raw_iid(thisRating))
                otherMovieID = int(trainset.to_raw_iid(otherRating))
                #genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                #yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                #mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
                similarities[thisRating, otherRating] = self.computeProductsSimilarity(thisMovieID, otherMovieID, data)
                #genreSimilarity + supplierSim *0.5
                similarities[otherRating, thisRating] = similarities[thisRating, otherRating]
        return similarities