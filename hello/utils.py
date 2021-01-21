from django.conf import settings
from math import sqrt, log
import numpy as np

from .models import Arm, DuelRecord, Counter

def dts_pick_first_arm(num_arms):
    upper_conf_bound = np.zeros((num_arms, num_arms))
    lower_conf_bound = np.zeros((num_arms, num_arms))
    duel_history = DuelRecord.objects.all()
    timestep = Counter.objects.all()[0]

    estimated_samples = np.zeros((num_arms, num_arms))

    for duel in duel_history:
        i = duel.first_arm - 1
        j = duel.second_arm - 1
        wins = duel.first_arm_wins
        losses = duel.second_arm_wins
            
        if wins + losses == 0:
            history = 1
            cb = 1
        else:
            history = wins / (wins + losses)
            cb = sqrt((settings.ALPHA * log(timestep.ts_count)) / (wins + losses))

        upper_conf_bound[i][j] = history + cb
        lower_conf_bound[i][j] = history - cb

        alpha = wins + 1
        beta = losses + 1

        estimated_samples[i][j] = np.random.beta(alpha, beta)
        estimated_samples[j][i] = 1 - estimated_samples[i][j]


    copeland_ub = (1 / (num_arms - 1)) * np.sum(
            upper_conf_bound, axis=1
    )
    candidates = np.argwhere(copeland_ub == np.amax(copeland_ub))

    likely_wins = np.zeros((num_arms, num_arms))
    for c in candidates:
        i = c[0]
        for j in range(num_arms):
            if i == j:
                continue
            if estimated_samples[i][j] > 1 / 2:
                likely_wins[i][j] = 1

    action = np.random.choice(
        np.argwhere(likely_wins == np.amax(likely_wins))[0]
    )  # break ties randomly
    
    return action + 1, lower_conf_bound


def dts_pick_second_arm(num_arms, first_action, lower_conf_bound):

    duel_history = DuelRecord.objects.all()

    expected_samples = np.zeros((num_arms, num_arms))
    for duel in duel_history:
        if duel.first_arm != first_action and duel.second_arm != first_action:
            continue
        i = duel.second_arm
        
        alpha = duel.first_arm_wins + 1
        beta = duel.second_arm_wins + 1
        
        expected_samples[i - 1][first_action - 1] = np.random.beta(alpha, beta)

    uncertain_pairs = np.zeros((num_arms, 1))
    for i in range(num_arms):
        if i == first_action - 1:
            continue
        if lower_conf_bound[i][first_action - 1] <= 1 / 2:
            uncertain_pairs[i] = expected_samples[i][first_action - 1]

    action = np.argmax(uncertain_pairs)
    return action + 1
