from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.db.models import F
import numpy as np
import random

from .models import Profile, Arm, Log, DuelRecord, Counter
from .forms import UserCreationForm
from .utils import dts_pick_first_arm, dts_pick_second_arm

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
    if request.method == 'POST':
        profile = Profile.objects.get(user=request.user)
        profile.comments = request.POST.get('comments')
        profile.save()
        logout(request)
        return render(request, "logout.html")
    else:
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

    # demo_dict = {'name': 'Demographics Survey', 'type': 'demo'}
    # if not profile.demo_completed:
    #     context['available'].append(demo_dict)
    # else:
    #     context['completed'].append(demo_dict)

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
    request.session['num_arms'] = all_arms.count()

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
        
        first_action, lower_conf_bound = dts_pick_first_arm(request.session['num_arms'])
        second_action = dts_pick_second_arm(request.session['num_arms'], first_action, lower_conf_bound)

        context['image1'] = first_action
        context['image2'] = second_action

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

    return render(request, "bandit.html", context)


def input(request, choice):
    context = {}

    survey_type = request.session.get('survey_type', 'uniform')
    sampling_code = "ts" if survey_type == "ts" else "un"
    all_arms = Arm.objects.all()

    first_arm = Arm.objects.get(img_id=request.session.get('first_arm_id'))
    second_arm = Arm.objects.get(img_id=request.session.get('second_arm_id'))
    try:
        duel_record = DuelRecord.objects.get(first_arm=request.session.get('first_arm_id'), second_arm=request.session.get('second_arm_id'))
    except:
        duel_record = DuelRecord.objects.get(first_arm=request.session.get('second_arm_id'), second_arm=request.session.get('first_arm_id'))

    new_log = Log(
        user=request.user,
        first_arm=first_arm.img_id,
        second_arm=second_arm.img_id,
        sampling_type=sampling_code,
        choice=choice)
    new_log.save()

    # Update and check the progress of the survey
    profile = Profile.objects.get(user=request.user)
    
    if survey_type == 'uniform':
        if choice == 1:
            first_arm.uniform_yes = F('uniform_yes') + 1
            second_arm.uniform_no = F('uniform_no') + 1

        elif choice == 2:
            first_arm.uniform_no = F('uniform_no') + 1
            second_arm.uniform_yes = F('uniform_no') + 1
        
        profile.uniform_images_seen  = F('uniform_images_seen') + 1

        if profile.uniform_images_seen >= 50:
            profile.uniform_completed = True
            profile.save()
            return redirect('profile')
    
    elif survey_type == 'ts':
        if choice == 1:
            first_arm.ts_yes = F('ts_yes') + 1
            second_arm.ts_no = F('ts_no') + 1
            duel_record.first_arm_wins = F('first_arm_wins') + 1
        elif choice == 2:
            first_arm.ts_no = F('ts_no') + 1
            second_arm.ts_yes = F('ts_yes') + 1
            duel_record.second_arm_wins = F('second_arm_wins') + 1
        profile.ts_images_seen = F('ts_images_seen') + 1

        if int(profile.ts_images_seen) >= 50:
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
        counter = Counter.objects.get_or_create()
        counter.uniform_count = F('uniform_count') + 1
        counter.total_count = F('total_count') + 1
        counter.save()

    if survey_type == 'ts':
        # Do uniform random sampling
        random_choice = random.sample(list(all_arms), 2)
        context['image1'] = random_choice[0].filename
        context['image2'] = random_choice[1].filename

        # Store the ID's of the two arms chosen
        request.session['first_arm_id'] = random_choice[0].img_id
        request.session['second_arm_id'] = random_choice[1].img_id

        counter.ts_count = F('ts_count') + 1
        counter.total_count = F('total_count') + 1
        counter.save()

    return render(request, "bandit.html", context)


def debrief(request):
    # Generate mturk code
    
    rand_num = random.randint(9999, 100000)
    code = request.user.username + str(rand_num)

    profile = Profile.objects.get(user=request.user)
    profile.code = code

    context = {'mturkcode': code}

    return render(request, "debrief.html", context)
