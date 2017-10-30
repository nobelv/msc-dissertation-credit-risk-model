# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Basic Securities functions - ignoring any jump process

import math
import numpy as np
import scipy.stats as sp
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
    :return: the value of the option
    """

    asset =  mf.big_a_0() * math.exp(mf.v_star() + 0.5 * mf.sigma()**2) * s
    normal_dist = np.random.normal(mf.h1(mf.v_bar(),s))

    price = asset * normal_dist

    return price

def pseudo_dig_call(s):
    """
    Similar to the asset or nothing call except that the payoff is now the monetary unit instead of the underlying.
    As the expected payoff is not discounted, Dig(s) corresponds to the probability of the firm surviving up to time s.

    :param s: time
    :return: the probability of the firm surviving up to time s
    """

    dig = 1 - sp.norm.cdf(mf.h1(mf.v_bar(), s), loc = 0.0, scale = 1.0) - mf.big_r_2_a * sp.norm.cdf(
        mf.h2(mf.v_bar(),s, loc =0.0, scale =1.0))

    return dig

def digital_down_and_out(s):
    """
    Non-deferable rebate of a put down-and-out with maturity s, exercise price and barrier equal to v_bar and rebate
    equal to 1 that only pays off if a negative jump does not occur up to maturity.
    We denote var_pi as var_pi = -1 (r - lambda_bar), as this model uses no jumps so var_pi is equal to -r

    Additionally the probability of this option ending up in the money is zero and thus the value of this option
    comes exclusively from the rebate.

    :param s: time
    :return: the value of the option
    """

    dig_hit =