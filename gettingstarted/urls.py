from django.urls import path, include
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.conf import settings
import hello.views

admin.autodiscover()


# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", hello.views.index, name="index"),
    path("login/", hello.views.user_login, name="user_login"),
    path("logout/", hello.views.user_logout, name='user_logout'),
    path("register/", hello.views.user_register, name="user_register"),
    path("profile/", hello.views.profile, name='profile'),
    path(r'(?P<survey_type>\w+)/$', hello.views.info, name="info"),
    path("bandit/", hello.views.bandit, name="bandit"),
    path('input/<int:choice>/', hello.views.input, name="input"),
    path("admin/", admin.site.urls),
]
