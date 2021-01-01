from django.conf import settings
from math import sqrt, log
import numpy as np

from .models import Arm, DuelRecord, Counter

def dts_pick_first_arm(num_arms):
    upper_conf_bound = np.zeros((num_arms, num_arms))
    lower_conf_bound = np.zeros((num_arms, num_arms))
    duel_history = DuelRecord.objects.select_related()
    timestep = Counter.objects.all()[0]

    for i in range(num_arms):
        for j in range(num_arms):
            if j == i:
                continue
            else:
                duel_log = duel_history.filter(first_arm=i + 1, second_arm=j + 1).first()
                if duel_log is not None:
                    wins = duel_log.first_arm_wins
                    losses = duel_log.second_arm_wins

                    reverse_order = False
                else:
                    duel_log = duel_history.filter(first_arm=j + 1, second_arm=i + 1).first()
                    wins = duel_log.second_arm_wins
                    losses = duel_log.first_arm_wins
                    
                    reverse_order = True

            if wins + losses == 0:
                history = 1
                cb = 1
            else:
                history = wins / (wins + losses)
                cb = sqrt((settings.ALPHA * log(timestep)) / (wins + losses))

            upper_conf_bound[i][j] = history + cb
            lower_conf_bound[i][j] = history - cb

    copeland_ub = (1 / (num_arms - 1)) * np.sum(
            upper_conf_bound, axis=1
    )
    candidates = np.argwhere(copeland_ub == np.amax(copeland_ub))

    estimated_samples = np.zeros((num_arms, num_arms))
    for i in range(num_arms):
        for j in range(i + 1, num_arms):
            if reverse_order:
                alpha = duel_log.second_arm_wins + 1
                beta = duel_log.first_arm_wins + 1
            else:
                alpha = duel_log.first_arm_wins + 1
                beta = duel_log.second_arm_wins + 1
            estimated_samples[i][j] = np.random.beta(alpha, beta)
            estimated_samples[j][i] = 1 - estimated_samples[i][j]

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

    duel_history = DuelRecord.objects.select_related()

    expected_samples = np.zeros((num_arms, num_arms))
    for i in range(num_arms):
        if i == first_action:
            continue
        else:
            duel_log = duel_history.filter(first_arm=first_action + 1, second_arm=i + 1).first()
            if duel_log is not None:
                alpha = duel_log.first_arm_wins + 1
                beta = duel_log.second_arm_wins + 1

            else:
                duel_log = duel_history.filter(first_arm=i + 1, second_arm=first_action + 1).first()
                alpha = duel_log.second_arm_wins + 1
                beta = duel_log.first_arm_wins + 1
            
            expected_samples[i][first_action] = np.random.beta(alpha, beta)

    uncertain_pairs = np.zeros((num_arms, 1))
    for i in range(num_arms):
        if i == first_action:
            continue
        if lower_conf_bound[i][first_action] <= 1 / 2:
            uncertain_pairs[i] = expected_samples[i][first_action]

    action = np.argmax(uncertain_pairs)
    return action + 1
