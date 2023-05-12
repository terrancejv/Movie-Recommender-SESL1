#Description: Build a move recommendation engine using 

#Import the libraries
import pandas as pd
import numpy as np
import gradio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def MovieRecommendation(Movie):
  try:
    #Load the data IMdb 2006 to 2016
    #Store the data
    df = pd.read_csv('IMDB-Movie-Data.csv')
    df['Movie_id'] = range(0,1000)

    #Show the first 3 rows of data
    df.head(3)

    #Get a count of the number of rows/movies in the data set and the number of columns
    df.shape

    #Create a list of important columns for the recommendation engine 
    columns = ['Actors', 'Director', 'Genre', 'Title']

    #Show the data
    df[columns].head(3)

    #Check for any missing values in the important columns
    df[columns].isnull().values.any()

    #Create a function to combine the values of the important columns into a single string
    def get_important_features(data):
      important_features = []
      for i in range(0, data.shape[0]):
        important_features.append(data['Actors'][i]+' '+data['Director'][i]+' '+data['Genre'][i]+' '+data['Title'][i])
      
      return important_features

    #Create a column to hold the combined strings
    df['important_features'] = get_important_features(df)

    #Show the data
    df.head(3)

    #Convert the text to a matrix of token counts
    cm = CountVectorizer().fit_transform(df['important_features'])

    #Get the cosine similarity matrix from the count matrix
    cs = cosine_similarity(cm)

    #Get the shape of the cosine similarity matrix
    cs.shape

    #Get the title of the movie that the user likes
    title = str(Movie)

    #Find the movies id
    movie_id = df[df.Title == title]['Movie_id'].values[0]

    #Create a list of enumerations for the similarity score [ (movie_id, similarity score), (...) ]
    scores = list(enumerate(cs[movie_id]))

    #Sort the list
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
    sorted_scores = sorted_scores[1:]

    #Create a loop to print the first 7 similar movies
    response = 'The 10 most recommended movies to ' + Movie + ' are:\n'
    j = 0
    for item in sorted_scores:
      movie_title = df[df.Movie_id == item[0]]['Title'].values[0]
      recommended_movie = '  ' + str(j+1) + ' - ' + movie_title + '\n'
      response += recommended_movie
      j = j+1
      if j>9:
        break
    return response

  #catches the index error if there are no movies found from the user's search
  except IndexError as e:
    raise gradio.Error("Search not found. Remember that the search is case-sensitive.")

demo = gradio.Interface(
  fn=MovieRecommendation, 
  inputs = "text", 
  outputs = "text", 
  title = "Movie Recommender", 
  description = "Find movie recommendations that are similar to the movie that is entered. \
  The search is case-sensitive, so you must enter the exact movie title. \
  \nExamples of movies (2006-2016) to search - The Dark Knight, The Wolf of Wall Street, Sausage Party"
)

demo.launch(share=True)

