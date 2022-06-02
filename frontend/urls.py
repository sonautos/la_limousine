from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('add_user/', views.addUser, name='register'),
    path('choosemovies/', views.chooseMovies, name='choosemovies'),
    path('choosecharacters/', views.chooseCharacters, name='choosecharacters'),
    path('movieslist/', views.moviesProposition, name='movieslist'),
    path('recomovies/', views.recoMovies, name='recomovies'),
    path('moviedetail/<int:id>', views.movieDetail, name='moviedetail'),
]