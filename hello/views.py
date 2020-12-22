from django.shortcuts import render
from django.http import HttpResponse
import random

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

def sample_uniform(request):
    if request.method == "GET":
        all_arms = arms.objects.all()
        random_choice = random.sample(all_arms, 2)
        return render(request, "index.html", {"choice1": random_choice[0].imgID, "choice2": random_choice[1].imgID})
    
