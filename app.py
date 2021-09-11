
import numpy as np
import pandas as pd
import flask
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


app = flask.Flask(__name__, template_folder='Templates')


# read the data
#bollywood = pd.read_csv("Bollywood_movies.csv")
#kollywood = pd.read_csv("Kollywood_movies.csv")
#mollywood = pd.read_csv("Mollywood_movies.csv")
#tollywood = pd.read_csv("Tollywood_movies.csv")
#sandalwood = pd.read_csv("Sandalwood_movies.csv")
#new_list = [bollywood,kollywood,mollywood,tollywood,sandalwood]
#all_movies = pd.concat(new_list,ignore_index = True,sort = False)

all_movies = pd.read_csv("movies.csv")

df = all_movies.reset_index()
indices = pd.Series(df.index, index=df['Title'])
all_titles = [df['Title'][i] for i in range(len(df['Title']))]
df = df.fillna("")
df["Comb"] = df['Cast']+' '+df['Title'] +' '+ df['Director']+' '+df['Genre']+' '+df['Music'] +' '+ df['language']

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['Comb'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_movies(movie_name):
    idx = indices[movie_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    movies = df.iloc[movie_indices][['Title']]
    return movies


@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
    if m_name not in all_titles:
        return(flask.render_template('no.html'))
    else:
        result = recommend_movies(m_name)
        title = []
        for i in range(len(result)):
            title.append(result.iloc[i][0])
        return flask.render_template('yes.html', movie_names=title,search_name=m_name)

if __name__ == '__main__':
    app.run(debug=True)


