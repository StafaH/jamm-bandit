from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
#from django.contrib.auth.forms import UserCreationForm
import random

from .models import Arm, User
from .forms import UserCreationForm

# Create your views here.


def index(request):
    return render(request, "index.html")


def user_register(request):
    if request.user.is_authenticated:
        return redirect('bandit')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('bandit')
        else:
            return render(request, 'register.html', {'form': form})
    else:
        form = UserCreationForm()
        return render(request, 'register.html', {'form': form})


def user_login(request):
    if request.user.is_authenticated:
        return render(request, 'bandit.html')
    if request.method == 'POST':
        # Process the request if posted data are available
        username = request.POST['username']
        password = request.POST['password']
        # Check username and password combination if correct
        user = authenticate(username=username, password=password)
        if user is not None:
            # Save session as cookie to login the user
            login(request, user)
            # Success, now let's login the user.
            return render(request, "bandit.html")
        else:
            # Incorrect credentials, let's throw an error to the screen.
            return render(request, "login.html", {'error_message': 'Incorrect username and / or password.'})
    else:
        # No post data availabe, let's just show the page to the user.
        return render(request, "login.html")


def user_logout(request):
    logout(request)
    return render(request, "logout.html")


def bandit(request):
    return render(request, "bandit.html")


def sample_uniform(request):
    if request.method == "GET":
        all_arms = Arm.objects.all()
        random_choice = random.sample(all_arms, 2)
        return {"choice1": random_choice[0].img_id,
                "choice2": random_choice[1].img_id}


def sample_DTS(request):
    raise NotImplementedError


def sample(request):
    if random.uniform(0, 1) < 0.5:
        return render(request, "index.html", sample_uniform(request))
    else:
        return render(request, "index.html", sample_DTS(request))
