from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
import random

from .models import Profile, Arm, Log
from .forms import UserCreationForm

# Create your views here.


def index(request):
    return render(request, "index.html")


def user_register(request):
    if request.user.is_authenticated:
        return redirect('profile')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)

            # Create a profile and history for this user
            profile = Profile(user=request.user)
            profile.save()

            return redirect('profile')
        else:
            return render(request, 'register.html', {'form': form})
    else:
        form = UserCreationForm()
        return render(request, 'register.html', {'form': form})


def user_login(request):
    if request.user.is_authenticated:
        return redirect('profile')
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
            return redirect('profile')
        else:
            # Incorrect credentials, let's throw an error to the screen.
            return render(request, "login.html", {'error_message': 'Incorrect username and / or password.'})
    else:
        # No post data availabe, let's just show the page to the user.
        return render(request, "login.html")


def user_logout(request):
    logout(request)
    return render(request, "logout.html")


def profile(request):
    # Grab the profile data to render
    profile = Profile.objects.get(user=request.user)

    context = {}
    context['available'] = []
    context['completed'] = []

    uniform_dict = {'name': 'Uniform Survey', 'type': 'uniform'}
    if not profile.uniform_completed:
        context['available'].append(uniform_dict)
    else:
        context['completed'].append(uniform_dict)

    ts_dict = {'name': 'Thompson Sampling Survey', 'type': 'ts'}
    if not profile.ts_completed:
        context['available'].append(ts_dict)
    else:
        context['completed'].append(ts_dict)

    demo_dict = {'name': 'Demographics Survey', 'type': 'demo'}
    if not profile.demo_completed:
        context['available'].append(demo_dict)
    else:
        context['completed'].append(demo_dict)

    return render(request, "profile.html", context)


def info(request, survey_type):
    context = {}

    if survey_type == 'uniform':
        context['info_text'] = 'Two faces will be shown to you. Please choose \
        the face that is more dominant. Once you have chosen, \
        another face will be shown. There are a total of 50 rounds'

        request.session['survey_type'] = 'uniform'
        return render(request, "info.html", context)

    elif survey_type == 'ts':
        context['info_text'] = 'Two faces will be shown to you. Please choose \
        the face that is more dominant. Once you have chosen, \
        another face will be shown. There are a total of 50 rounds'
        request.session['survey_type'] = 'ts'
        return render(request, "info.html", context)

    elif survey_type == 'demo':
        return render(request, "demographics.html", context)


def bandit(request):
    context = {}

    survey_type = request.session.get('survey_type', 'uniform')
    all_arms = Arm.objects.all()

    if survey_type == 'uniform':
        # Do uniform random sampling
        random_choice = random.sample(list(all_arms), 2)
        context['image1'] =  random_choice[0].filename
        context['image2'] = random_choice[1].filename

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

    if survey_type == 'ts':
        # Do uniform random sampling
        random_choice = random.sample(list(all_arms), 2)
        context['image1'] = random_choice[0].filename
        context['image2'] = random_choice[1].filename

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

    return render(request, "bandit.html", context)


def input(request, choice):
    context = {}

    survey_type = request.session.get('survey_type', 'uniform')
    all_arms = Arm.objects.all()

    first_arm = Arm.objects.get(img_id=request.session.get('first_arm_id'))
    second_arm = Arm.objects.get(img_id=request.session.get('second_arm_id'))

    new_log = Log(
        user=request.user,
        first_arm=first_arm.img_id,
        second_arm=second_arm.img_id,
        choice=choice)
    new_log.save()

    # Update and check the progress of the survey
    profile = Profile.objects.get(user=request.user)
    
    if survey_type == 'uniform':
        if choice == 1:
            first_arm.uniform_yes += 1
            second_arm.uniform_no += 1
        elif choice == 2:
            first_arm.uniform_no += 1
            second_arm.uniform_yes += 1
        
        profile.uniform_images_seen += 1

        if profile.uniform_images_seen >= 50:
            profile.uniform_completed = True
            profile.save()
            return redirect('profile')
    
    elif survey_type == 'ts':
        if choice == 1:
            first_arm.ts_yes += 1
            second_arm.ts_no += 1
        elif choice == 2:
            first_arm.ts_no += 1
            second_arm.ts_yes += 1
        profile.ts_images_seen += 1

        if profile.ts_images_seen >= 50:
            profile.ts_completed = True
            profile.save()
            return redirect('profile')

    first_arm.save()
    second_arm.save()
    profile.save()

    if survey_type == 'uniform':
        # Do uniform random sampling
        random_choice = random.sample(list(all_arms), 2)
        context['image1'] = random_choice[0].filename
        context['image2'] = random_choice[1].filename

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

    if survey_type == 'ts':
        # Do uniform random sampling
        random_choice = random.sample(list(all_arms), 2)
        context['image1'] = random_choice[0].filename
        context['image2'] = random_choice[1].filename

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

    return render(request, "bandit.html", context)
