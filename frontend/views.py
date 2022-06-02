from django.shortcuts import render
import config
import numpy as np
import pandas as pd
import random
import datetime as datetime
from sklearn.neighbors import NearestNeighbors
from frontend.tmdb_request import cast_from_tconst, infos_from_id, yt_from_id
import aiohttp
import asyncio
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import copy

api_key = config.api_key

df = pd.read_csv("./movies/top_movies.tsv", sep='|', lineterminator='\n', na_values="\\N")
df_ml = pd.read_csv("./movies/df_ml.csv", na_values="\\N")
df_matching = pd.read_csv('./movies/df_matching.csv', na_values="\\N")

genres_list = ['Action',
        'Fantasy', 'Film-Noir', 'Romance', 'War', 'Sport', 'Sci-Fi',
        'Documentary', 'Biography', 'Comedy', 'Western', 'Mystery', 'Music',
        'Family', 'Horror', 'Animation', 'Thriller', 'History', 'Adventure',
        'Musical', 'Crime', 'Drama']


def my_own_get_dummies(id, df, col):
    # Create genres list
    if col == 'genres':
        my_set = set()
        my_list = [id]
        for i in df[col]:
            my_set.add(i)
        for i in my_set:
            my_list.extend(i.split(','))
        res = list(set(my_list))
    else: 
        res = ['actor', 'actress', 'producer', 'director', 'writer']

    # Create a DataFrame with the genre list and fill it with tconst & genres values
    tmp_df = pd.DataFrame(columns=res)
    tmp_df[id] = df[id]
    tmp_df[col] = df[col]
    tmp_df = tmp_df.set_index(id)

    # Fill with 1 each columns
    for i in tmp_df.columns:
        if i != col:
            tmp_df[i][tmp_df[col].str.contains(i)] = 1
            tmp_df[i] = tmp_df[i].fillna(0)
    tmp_df.drop(columns=col, inplace=True)
    tmp_df = df.merge(tmp_df, how='left', on=id)
    tmp_df = tmp_df.drop(columns=[col])
    return tmp_df

def index(request):
    return render(request, 'index.html', {})

# Reco sans paramètre.
def recoMovies(request):
    title = request.POST.getlist('browser')
    user = request.POST['user']
    user = eval(user)

    title = title[0]
    
    ml = df_ml[df_ml['title'] == title]

    tconst = ml.tconst
    tconst = tconst
    reco_list = []
    reco_dict = {}
    nb_neighbors = 30
    
    col_name = ['Fantasy', 'History', 'Sci-Fi', 'Western', 'Horror',
            'Music', 'News', 'Animation', 'Sport', 'Family', 'Mystery', 'Musical','Comedy', 'Thriller', 'Crime', 'Documentary', 'War', 'Romance',
            'Adventure', 'Drama', 'Action', 'Biography', 'region_num', 'category_num','nconst_num','WR']

    X = df_ml[col_name]    
    reco_list = []
    reco_dict = {}

    distanceKNN = NearestNeighbors(n_neighbors=8).fit(X)
    reco = distanceKNN.kneighbors(df_ml.loc[df_ml['tconst'] == tconst.iloc[0], col_name])
    for j in range(len(reco[1])):
        index = reco[1][j]
        distance = reco[0][j]
        for id, dist in zip(index, distance):
            if dist < 1.6:
                reco_dict[df_ml.iloc[id]['tconst']] = dist
    del reco_dict[tconst.iloc[0]]
    reco_dict_copy = copy.copy(reco_dict)
    for i in reco_dict_copy:
        distanceKNN = NearestNeighbors(n_neighbors=8).fit(X)
        reco = distanceKNN.kneighbors(df_ml.loc[df_ml['tconst'] == i, col_name])
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                if dist < 1.1:
                    reco_dict[df_ml.iloc[id]['tconst']] = dist
    reco_dict_copy = copy.copy(reco_dict)
    for i in reco_dict_copy:
        distanceKNN = NearestNeighbors(n_neighbors=8).fit(X)
        reco = distanceKNN.kneighbors(df_ml.loc[df_ml['tconst'] == i, col_name])
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                if dist < 1.1:
                    reco_dict[df_ml.iloc[id]['tconst']] = dist
    if len(reco_dict) < 5:
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                if dist < 2:
                    reco_dict[df_ml.iloc[id]['tconst']] = dist
        try:
            del reco_dict[tconst.iloc[0]]
        except:
            pass
        reco_dict_copy = copy.copy(reco_dict)
        for i in reco_dict_copy:
            distanceKNN = NearestNeighbors(n_neighbors=8).fit(X)
            reco = distanceKNN.kneighbors(df_ml.loc[df_ml['tconst'] == i, col_name])
            for j in range(len(reco[1])):
                index = reco[1][j]
                distance = reco[0][j]
                for id, dist in zip(index, distance):
                    if dist < 1.6:
                        reco_dict[df_ml.iloc[id]['tconst']] = dist
    reco_dict = dict(sorted(reco_dict.items(), key=lambda x: x[1]))
    reco_list = [i for i in reco_dict.keys() if i != tconst.iloc[0]]
    # user['movies_list'].append(tconst.iloc[0])
    all_list = df_ml['title'].drop_duplicates().to_dict()
    reco_movies = asyncio.run(get_tconsts_infos(reco_list))
    

    if df.tconst.isin([tconst.iloc[0]]).any().any():
        if tconst.iloc[0] not in user['movies_list']:
            tconst_unique = list(set(tconst))
            user['movies_list'].extend(tconst_unique)

    search_movie = []
    search_movie.append(tconst.iloc[0])
    movie = asyncio.run(get_tconsts_infos(search_movie))
    
    return render(request, 'recomovies.html', {
                                                'reco_movies': reco_movies, 
                                                'user': user, 
                                                'title': title, 
                                                'all_list': all_list, 
                                                'movie': movie
                                                })

def movieDetail(request, id):
    id = id
    res = infos_from_id(id)
    video = yt_from_id(id)
    casts = cast_from_tconst(id)
    tconst = res['imdb_id']
    title = res['title']
    genres = [i['name'] for i in res['genres']]
    resume = res['overview']
    date = res['release_date']
    poster = 'https://image.tmdb.org/t/p/original' + res['poster_path']
    runtime = res['runtime']
    vote_average = res['vote_average']
    
    global df_matching
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df_matching['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    try:
        df_matching = df_matching.reset_index()
    except:
        pass
    indices = pd.Series(df_matching.index, index=df_matching['tconst'])
    def get_recommendations(title, cosine_sim=cosine_sim2):
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df_matching['tconst'].iloc[movie_indices]
    
    try:
        reco_5 = get_recommendations(tconst)
    except:
        reco_5 = []
    reco_5 = list(reco_5)
    if tconst in reco_5:
        reco_5.remove(tconst)
    movies_list =asyncio.run(get_tconsts_infos(reco_5))

    return render(request, 'moviedetail.html', {'title': title, 
                                                'genres': genres, 
                                                'resume': resume,
                                                'date' : date,
                                                'tconst': tconst,
                                                'poster': poster, 
                                                'results': res, 
                                                'video': video, 
                                                'casts': casts,
                                                'range5': range(5),
                                                'runtime' : runtime,
                                                'vote_average' : vote_average, 
                                                'reco_5': movies_list
                                                })

def addUser(request):
    email = request.POST['email']
    return render(request, 'add_user.html', {'email': email})

def chooseMovies(request):
    email = request.POST['email']
    username = request.POST['username']
    bdate = request.POST['bdate']
    password1 = request.POST['password1']
    
    byear = datetime.datetime.strptime(bdate, '%Y-%m-%d').year
    age = round(datetime.datetime.now().year - byear, 0)
        
    df_people = df[['tconst', 'nconst', 'name', 'french_title', 'byear', 'job', 'startyear','runtimeminutes', 'averagerating', 'numvotes','wr_score', 'genres', 'original_lang']].copy()
    df_people.job = df_people.job.apply(lambda x: x.replace("', '", ",").replace("['", "").replace("']", ""))
    df_people['original_lang'] = df_people['original_lang'].factorize()[0]
    df_people = my_own_get_dummies('tconst', df_people, 'job')
    df_people = my_own_get_dummies('tconst', df_people, 'genres')
    df_people = df_people.drop_duplicates(['tconst'])
    movies_per_year = {}
    def user_age(param):
        df_users = pd.DataFrame(columns=df_people.columns)
        user = {
            'tconst' : 'tt00001',
            'nconst' : 'nn00001',
            'name' : username,
            'french_title' : email,
            'byear' : byear,
            'startyear' : param,
            'runtimeminutes' : df_people.runtimeminutes.median(),
            'averagerating' : df_people.averagerating.quantile(.75),
            'numvotes' : df_people.numvotes.median(),
            'wr_score' : df_people.wr_score.quantile(.75),
            'original_lang' : 1,
            'actor' : .5,
            'actress' : .5,
            'producer' : .5,
            'director' : .5,
            'writer' : .5,
            'Sci-Fi' : 0.045,
            'Animation' : 1,
            'Music' : 0.045,
            'War' : 0.045,
            'Biography' : 0.045,
            'Romance' : 0.045,
            'History' : 0.045,
            'Documentary' : 0.045,
            'Crime' : 0.045,
            'Fantasy' : 0.045,
            'Family' : 1,
            'Western' :0.045,
            'Adventure' : 0.045,
            'Horror' : 0.045,
            'Comedy' : 0.045,
            'Musical' : 0.045,
            'Sport' : 0.045,
            'Film-Noir' : 0.045,
            'Action' : 0.045,
            'Mystery' : 0.045,
            'Drama' : 0.045,
            'Thriller' : 0.045,
        }
        df_users.loc[0] = user.values()
        x_temp = pd.concat([df_people, df_users], ignore_index=True)
        col_name = ['startyear','original_lang','averagerating', 'Action',
                    'Fantasy', 'Film-Noir', 'Romance', 'War', 'Sport', 'Sci-Fi',
                    'Documentary', 'Biography', 'Comedy', 'Western', 'Mystery', 'Music',
                    'Family', 'Horror', 'Animation', 'Thriller', 'History', 'Adventure',
                    'Musical', 'Crime', 'Drama']
        user_tconst = x_temp.iloc[-1:]['tconst'].values[0]
        X = x_temp[col_name]
        distanceKNN = NearestNeighbors(n_neighbors=30).fit(X)
        reco = distanceKNN.kneighbors(x_temp.loc[x_temp['tconst'] == user_tconst, col_name])
        reco_list = []
        reco_dict = {}
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            reco_list.extend(list(set(x_temp.iloc[index]['tconst'])))
            for id, dist in zip(index, distance):
                reco_dict[x_temp.iloc[id]['tconst']] = dist
        del reco_dict['tt00001']
        return reco_dict
    

    for i in range(0, 11):
        try:
            search_year = (round(datetime.datetime.now().year - ((age)/i), 0))
            movies_per_year.update(user_age(search_year))
        except:
            search_year = (datetime.datetime.now().year)
            movies_per_year.update(user_age(search_year))
            
    # Classement par distance
    movies_per_year = dict(sorted(movies_per_year.items(), key=lambda x: x[1]))
    
    #
    movies_list = [i for i in movies_per_year.keys()]
    
    # Création d'un user:
    user = {'id' : '01',
            'username' : username, 
            'email' : email,
            'byear': bdate, 
            'password': password1,
            'movies_list' : []
            }
    
    # Frontend
    movies = asyncio.run(get_tconsts_infos(movies_list))  
    return render(request, 'choosemovies.html', {
                                                  'user' : user,
                                                  'movies': movies,
                                                })



def chooseCharacters(request):
    tconsts = request.POST.getlist('id')
    user = request.POST['user']
    user = eval(user)
    for i in user['movies_list']:
        tconsts.extend(i)
    
    # Machine Learning
    df_people = df[['tconst', 'nconst', 'name', 'french_title', 'byear', 'job', 'startyear','runtimeminutes', 'averagerating', 'numvotes','wr_score', 'genres', 'original_lang']].copy()
    df_people.job = df_people.job.apply(lambda x: x.replace("', '", ",").replace("['", "").replace("']", ""))
    df_people = my_own_get_dummies('tconst', df_people, 'job')
    df_people = my_own_get_dummies('tconst', df_people, 'genres')
    df_people = df_people.drop_duplicates(['tconst', 'nconst'])
    col_name = ['startyear','averagerating','Action',
        'Fantasy', 'Film-Noir', 'Romance', 'War', 'Sport', 'Sci-Fi',
        'Documentary', 'Biography', 'Comedy', 'Western', 'Mystery', 'Music',
        'Family', 'Horror', 'Animation', 'Thriller', 'History', 'Adventure',
        'Musical', 'Crime', 'Drama']

    X = df_people[col_name]
    distanceKNN = NearestNeighbors(n_neighbors=30).fit(X)

    reco_dict = {}
    for i in tconsts:
        reco = distanceKNN.kneighbors(df_people.loc[df_people['tconst'] == i, col_name])            
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                reco_dict[df_people.iloc[id]['nconst']] = dist

    tconsts = list(set(tconsts))
    user['movies_list'].extend(tconsts)
    reco_dict = dict(sorted(reco_dict.items(), key=lambda x: x[1]))
    movies_list = [i for i in reco_dict.keys()]
    persons = asyncio.run(get_nconsts_infos(movies_list))
    return render(request, 'choosecharacters.html', {
                                                    'tconsts' : tconsts, 
                                                    'persons': persons, 
                                                    'user': user
                                                    })
    
def moviesProposition(request):
    nconsts = request.POST.getlist('nconsts')
    user = request.POST['user']
    user = eval(user)
    
    if len(nconsts) > 1:
        tconsts = df[df.nconst.isin(nconsts)].sort_values('averagerating').tconst.head(3)
        tconsts = list(tconsts)
        movies_list = [i for i in user['movies_list']]
        tconsts.extend(movies_list)
    else: 
        tconsts = []
        movies_list = [i for i in user['movies_list']]
        tconsts.extend(movies_list)
    
    df_people = df[['tconst', 'nconst', 'name', 'french_title', 'byear', 'job', 'startyear','runtimeminutes', 'averagerating', 'numvotes','wr_score', 'genres', 'original_lang']].copy()
    df_people.job = df_people.job.apply(lambda x: x.replace("', '", ",").replace("['", "").replace("']", ""))
    df_people = my_own_get_dummies('tconst', df_people, 'job')
    df_people = my_own_get_dummies('tconst', df_people, 'genres')
    df_people = df_people.drop_duplicates(['tconst', 'nconst'])
    col_name = ['startyear', 'Action',
        'Fantasy', 'Film-Noir', 'Romance', 'War', 'Sport', 'Sci-Fi',
        'Documentary', 'Biography', 'Comedy', 'Western', 'Mystery', 'Music',
        'Family', 'Horror', 'Animation', 'Thriller', 'History', 'Adventure',
        'Musical', 'Crime', 'Drama']
    
    X = df_people[col_name]
    distanceKNN = NearestNeighbors(n_neighbors=30).fit(X)
    reco_list = []
    reco_dict = {}
    tconsts = tconsts[:-31:-1]
    for i in tconsts:
        reco = distanceKNN.kneighbors(df_people.loc[df_people['tconst'] == i, col_name])            
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                reco_dict[df_people.iloc[id]['tconst']] = dist
    reco_dict_copy = copy.copy(reco_dict)
    for i in reco_dict_copy:
        distanceKNN = NearestNeighbors(n_neighbors=10).fit(X)
        reco = distanceKNN.kneighbors(df_people.loc[df_people['tconst'] == i, col_name])
        for j in range(len(reco[1])):
            index = reco[1][j]
            distance = reco[0][j]
            for id, dist in zip(index, distance):
                if dist < 1.5:
                    reco_dict[df_people.iloc[id]['tconst']] = dist
    
    reco_dict = dict(sorted(reco_dict.items(), key=lambda x: x[1]))
    reco_list = [i for i in reco_dict.keys() if i not in tconsts]
    tconsts = list(set(tconsts))
    user['movies_list'].extend(tconsts)
    reco_movies = asyncio.run(get_tconsts_infos(reco_list))
    
    all_list = df_ml.sort_values('title').title.drop_duplicates().to_dict()
    
    return render(request, 'movieslist.html', {
                                                'reco_movies' : reco_movies, 
                                                'user': user, 
                                                'all_list' : all_list
                                                })

# TCONST

async def get_tconst(session, url, tc):
    async with session.get(url) as resp:
        try:
            result = await resp.json()
            title = result['movie_results'][0]['title']
            poster = result['movie_results'][0]['poster_path']
            overview = result['movie_results'][0]['overview']
            return {'tc': tc,
                    'result': result['movie_results'][0],
                    'title': title, 
                    'poster': f"https://image.tmdb.org/t/p/original{poster}", 
                    'overview': overview
                    }
        except:
            pass
        

async def get_tconsts_infos(tconsts):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for tc in tconsts:
            url = f'https://api.themoviedb.org/3/find/{ tc }?api_key={ api_key }&external_source=imdb_id&language=fr-FR'
            tasks.append(asyncio.ensure_future(get_tconst(session, url, tc)))

        movies = {}
        original_tconst = await asyncio.gather(*tasks)
        for tc in original_tconst:
            try:
                movies[tc['tc']] = tc
            except:
                pass
           

        return movies


# NCONST

async def get_id(session, url):
    async with session.get(url) as resp:
        try:
            result = await resp.json()
            id = result['person_results'][0]['id']
            return id
        except:
            pass
    
async def get_result(session, url2):
    async with session.get(url2) as resp:
        result = await resp.json()
        return result
    

async def get_nconsts_infos(nconsts):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for nconst in nconsts:
            url = f'https://api.themoviedb.org/3/find/{nconst}?api_key={api_key}&external_source=imdb_id'
            tasks.append(asyncio.ensure_future(get_id(session, url)))

        ids = []
        original_nconst = await asyncio.gather(*tasks)
        for id in original_nconst:
            ids.append(id)
        
        tasks2 = []
        for person_id in ids:
            url2 = f'https://api.themoviedb.org/3/person/{person_id}?api_key={api_key}&language=fr-FR'
            tasks2.append(asyncio.ensure_future(get_result(session, url2)))

        people = {}
        original_result = await asyncio.gather(*tasks2)
        for res in original_result:
            try:
                people[res['id']] = res
            except:
                pass
            
        return people
            
    # await get_tconsts_infos(tconsts)