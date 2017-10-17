# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Basic Securities functions - ignoring any jump process

import math
import numpy as np
import matplotlib

# Have to import the model_functions.py at some point to make functions work.
# As they use inputs from there.
import model_functions as mf


def asset_nothing_call(s):
    """
    A security that pays A_s if the firm is not closed at time s.
    Resembles a down-and-out call option with maturity s and strike 0 with barrier v_bar.
    Two differences, first the security only pays A_s if a negative jump has not occured at time s.
    Second the payoff is not discounted.

    :param s: time
    """
    asset =  mf.big_a_0() * math.exp(mf.v_star() + 0.5 * mf.sigma()**2) * s
    normal_dist = np.random.normal(mf.h1(mf.v_bar(),s))

    price = asset * normal_dist

    return price
