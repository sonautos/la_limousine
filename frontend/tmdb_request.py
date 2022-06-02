import requests
from aiohttp import ClientSession
import aiohttp
import asyncio
import time
api_key = 'f68d78c330babd6aae4a2cfde1eb0be6'

start_time = time.time()

def infos_from_tconst(tconst):
    response = requests.get('https://api.themoviedb.org/3/find/' +  tconst + '?api_key=' + api_key + '&external_source=imdb_id&language=fr-FR')
    return response.json() # store parsed json response

def get_nconst(tconst):
    response = requests.get('https://api.themoviedb.org/3/find/' +  tconst + '?api_key=' + api_key + '&external_source=imdb_id&language=fr-FR')
    response = response.json()
    pers_id = response['person_results'][0]['id']
    pers_infos = requests.get('https://api.themoviedb.org/3/person/' + str(pers_id) + '?api_key=' + api_key + '&language=fr-FR')
    pers_infos = pers_infos.json()
    return pers_infos

def get_nconst1(tconst):
    response = requests.get('https://api.themoviedb.org/3/find/' +  tconst + '?api_key=' + api_key + '&external_source=imdb_id&language=fr-FR')
    response = response.json()
    pers_id = response['person_results'][0]['id']
    pers_infos = requests.get('https://api.themoviedb.org/3/person/' + str(pers_id) + '?api_key=' + api_key + '&language=fr-FR')
    pers_infos = pers_infos.json()
    knowfor = []
    for i in response['person_results'][0]['known_for']:
        movie = i['original_title']
        knowfor.append(movie)
    knowfor
    return [knowfor, pers_infos]

def list_language(lang):
    response = requests.get('https://api.themoviedb.org/3/discover/movie?api_key=' + api_key + '&language=' + lang)
    return response.json()

def infos_from_id(id):
    response = requests.get('https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=' + api_key + '&language=fr-FR')
    return response.json()

def yt_from_id(id):
    try:
        response = requests.get('https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=' + api_key + '&append_to_response=release_dates,videos')
        response =  response.json()
        response = response['videos']['results'][0]
        return response
    except:
        return ''
    
def cast_from_tconst(id):
    response = requests.get('https://api.themoviedb.org/3/movie/' + str(id) + '?api_key=' + api_key + '&language=fr-FR&append_to_response=credits')
    response = response.json()
    response = response['credits']['cast']
    return response
            
# await get_tconsts_infos(tconsts)

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


