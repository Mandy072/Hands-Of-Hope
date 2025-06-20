from django.contrib import admin
from django.urls import path
from app import views


urlpatterns = [
    path("", views.index, name= "home"),
    path("log_in", views.log_in, name= "log_in"),
    path("register", views.register, name= "register"),
    path("dashboard", views.dashboard, name= "dashboard"),
    path("log_out", views.log_out, name= "log_out"),
    path("new_sign", views.new_sign, name= "new_sign"),
    path("capture_sign", views.capture_sign, name= "capture_sign"),
    path("learn_sign", views.learn_sign, name= "learn_sign"),
    path("delete/<str:sign_name>", views.delete, name= "delete"),
]