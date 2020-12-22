from django.shortcuts import render
from django.http import HttpResponse

from .models import arms, users

# Create your views here.
def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "index.html")


def db(request):

    arms = arms()
    arms.save()

    arms = arms.objects.all()

    return render(request, "db.html", {"arms": arms})
