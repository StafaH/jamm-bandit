from django.urls import path, include
from django.contrib import admin
from django.contrib.auth import views as auth_views
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
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("bandit/", hello.views.bandit, name="bandit"),
    path("db/", hello.views.db, name="db"),
    path("admin/", admin.site.urls),
]
