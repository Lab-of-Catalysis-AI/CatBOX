# -*- coding: utf-8 -*-
#==========================================
# Title:  probability.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

"""
Created based on
https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
"""

import random


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    # Handle edge cases where sum(weights) might be nan or 0
    total = sum(weights)
    if not isinstance(total, (int, float)) or total <= 0 or len(weights) == 0:
        # Fallback: return a random valid index
        return random.randint(0, len(weights) - 1) if len(weights) > 0 else 0

    choice = random.uniform(0, total)
    #    print(choice)
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

    # If we get here due to floating point precision issues, return the last valid index
    return len(weights) - 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribtuion toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0):

    # Handle edge cases
    if len(weights) == 0:
        return tuple()

    # Filter out nan and inf values, replace with small positive values
    cleaned_weights = []
    for w in weights:
        if isinstance(w, (int, float)) and w > 0 and w != float('inf'):
            cleaned_weights.append(w)
        else:
            cleaned_weights.append(1e-10)  # Small positive value

    theSum = float(sum(cleaned_weights))
    if theSum <= 0:
        # If sum is still <= 0, use uniform distribution
        uniform_prob = 1.0 / len(weights)
        return tuple(uniform_prob for _ in weights)

    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in cleaned_weights)


def mean(aList):
    theSum = 0
    count = 0

    for x in aList:
        theSum += x
        count += 1

    return 0 if count == 0 else theSum / count
