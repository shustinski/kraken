from django.urls import path

from . import views

app_name = 'webui'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('start/', views.start_processing, name='start_processing'),
    path('stop/', views.stop_processing, name='stop_processing'),
    path('api/status/', views.status_api, name='status_api'),
]
