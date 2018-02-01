# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Basic Securities functions - ignoring any jump process

import math
import scipy.stats as sp
import numpy as np
import model_functions as mf


def an(biga0, vstar, sigma, vbar, bigr, smalla, s):
    """
    Pseudo-asset or nothing "no liquidation" call.

    A security that pays A_s if the firm is not closed at time s.
    Resembles a down-and-out call option with maturity s and strike 0 with barrier v_bar.
    Two differences, first the security only pays A_s if a negative jump has not occured at time s.
    Second, the payoff is not discounted.

    :param biga0: The value of a security A_t at the beginning of the process.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param vbar: The barrier as calculated by derivatives of payout, coupon and capex functions.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.
    :param s: time.

    :return: the value of the option.
    """

    return biga0 * math.exp((vstar + 0.5 * sigma ** 2) * s) * \
        1 - sp.norm.cdf(mf.h3(biga0, vstar, sigma, vbar, s), loc=0.0, scale=1.0) - \
        bigr ** ((2 * smalla) + 2) * sp.norm.cdf(mf.h4(bigr, vbar, vstar, sigma, vbar, s), loc=0.0, scale=1.0)


def dig_s(vbar, bigr, smalla, biga0, vstar, sigma, s):
    """
    Pseudo-Digital "no liquidation" call.

    Similar to the asset or nothing call except that the payoff is now the monetary unit instead of the underlying.
    As the expected payoff is not discounted, Dig(s) corresponds to the probability of the firm surviving up to time s.

    :param vbar: The barrier as calculated by derivatives of payout, coupon and capex functions.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.
    :param biga0: The value of the security at time 0.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param s: time.

    :return: the probability of the firm surviving up to time s.
    """

    return 1 - sp.norm.cdf(mf.h1(biga0, vstar, sigma, vbar, s), loc=0.0, scale=1.0) - (bigr ** (2 * smalla)) * \
        sp.norm.cdf(mf.h2(bigr, vbar, vstar, sigma, vbar, s), loc=0.0, scale=1.0)


def dig_hit_s(varpi, sigma, vstar, bigr, smalla, s):
    """
    Digital down-and-out "no jump" put with rebate.

    Non-deferable rebate of a put down-and-out with maturity s, exercise price and barrier equal to v_bar and rebate
    equal to 1 that only pays off if a negative jump does not occur up to maturity.
    We denote var_pi as var_pi = -1 (r - lambda_bar), as this model uses no jumps so var_pi is equal to -r

    Additionally the probability of this option ending up in the money is zero and thus the value of this option
    comes exclusively from the rebate.

    :param varpi: The risk free rate * -1.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param vstar: The lognormal adjusted drift.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.
    :param s: time.

    :return: the value of the option.
    """

    return mf.big_f(varpi, (np.log(bigr) / sigma), vstar / sigma, s) + (bigr ** (2 * smalla)) * \
        mf.big_f(varpi, (np.log(bigr) / sigma), -(vstar / sigma), s)
