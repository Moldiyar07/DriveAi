"""
URL configuration for ml_model app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Главная страница
    path('predict/', views.predict, name='predict'),  # API для предсказания
]
