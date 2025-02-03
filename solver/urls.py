from django.urls import path
from . import views
urlpatterns=[
    path('',views.home,name='home'),
    path('solve',views.solve,name='solve'),
    path('solve_lp', views.solve_lp, name='solve_lp'),
]

