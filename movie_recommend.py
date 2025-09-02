from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import ast
import nltk
import pickle


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")

# print(movies.head(1))
# print(credits.head(1))

# print(movies.merge(credits,on='title').shape)
# # print(data)
# movies.info()

movies=movies.merge(credits,on='title')

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# print(movies.head())

movies.dropna(inplace=True)
# print(movies.dropna(inplace=True))

movies.duplicated().sum()
# print(movies.duplicated().sum())

movies.isnull().sum()
# print(movies.isnull().sum())

# print(movies.iloc[0].genres)


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(convert)
# print(movies['genres'])
# print(movies.head())

movies['keywords']=movies['keywords'].apply(convert)
# print(movies['keywords'])


def convert2(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            l.append(i['name'])
            counter=+1
        else:
            break    
    return l

movies['cast']=movies['cast'].apply(convert2)

# print(movies['cast'].head(1))

def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l

movies['crew']=movies['crew'].apply(fetch_director)

# print(movies['crew'].head())

       
movies['overview']=movies['overview'].apply(lambda x:x.split())

# print(movies['overview'].head(1))

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
print(movies['genres'].head(1),movies['cast'].head(1),movies['crew'].head(1),movies['keywords'].head(1))

movies['tags']=movies['overview']+movies['cast']+movies['crew']+movies['keywords']+movies['genres']

# print()
# print(movies['tags'].head(1))

new_df=movies[['movie_id','title','tags']].copy()

# print(new_df)

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

# print(new_df)

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

# print(new_df.head(0))

ps=PorterStemmer()

def stem(text):
    y=[]

    for i in text.split():
       y.append( ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)
# print(new_df.head(1))
     
cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()
# print(vectors[0])

cv.get_feature_names_out()
# print(cv.get_feature_names_out())

# print(cosine_similarity(vectors).shape)

similarity=cosine_similarity(vectors)
print(similarity[0])

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# print(recommend('Avatar'))

# pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))
