from django.shortcuts import render
from django.http import HttpResponse
import random

from .models import Arm, User

# Create your views here.


def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "index.html")


def login(request):
    return render(request, "login.html")


def bandit(request):
    return render(request, "bandit.html")


def db(request):

    arms = Arm()
    arms.save()

    arms = arms.objects.all()

    return render(request, "db.html", {"arms": arms})


def sample_uniform(request):
    if request.method == "GET":
        all_arms = Arm.objects.all()
        random_choice = random.sample(all_arms, 2)
        return {"choice1": random_choice[0].imgID,
                "choice2": random_choice[1].img_id}


def sample_DTS(request):
    raise NotImplementedError


def sample(request):
    if random.uniform(0, 1) < 0.5:
        return render(request, "index.html", sample_uniform(request))
    else:
        return render(request, "index.html", sample_DTS(request))
