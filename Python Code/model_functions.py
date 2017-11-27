# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

import numpy as np
import scipy.stats as sp
import mysql.connector
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


########################################################################################################################
# State variable functions - A_0, miu_A, sigma, small_r and miu_delta
########################################################################################################################

def big_a_0(delta0, miua, miudelta ):
    """
    The value of a security A_t at the beginning of the process is given by the function A0 = delta_0 / (miu_A - g).
    Where g = miu_delta - (lambda*jump), with jump being equal to 0 g = miu_delta.

    :param delta0: the value of EBITDA;
    :param miua: the discount rate, assumed to be constant for mathematical tractability (miu_big_a)
    :param miudelta: instantaneous growth rate of the firm
    """

    return  delta0 / (miua - miudelta)


def miu_big_a(r, mbar, sigma):
    """
    The discount rate, assumed to be constant

    :return: the discount rate
    """

    return r + mbar * sigma


def miu_delta(gvkey, ebitdadict, sigma):
    """
    Instantaneous growth rate of the firm. Essentially the historical growth rate of the state variable,
    which is EBITDA in our model. We use data from the MySQL database and estimate a growth rate through
    linear regressions.

    :param gvkey: gvkey corresponding to the firm
    :param ebitdadict: dictionary containing all gvkeys and their ebitda values
    :param sigma: sigma for company
    :return: List of lists containing the historical growth rate of each of the firms based on their EBITDA.
    """
    ebitdalist = ebitdadict[gvkey]

    return miudelta


def small_r():

    return

def func_sigma(gvkey, ebitdadict):
    """
    Lognormal return on EBITDA and the standard deviation of these returns.

    :return: List of lists containing the sigma estimate based on lognormal return of EBITDA per company
    """
    ebitda_list = ebitdadict[gvkey]
    logs = []

     # Skip element 1, no previous EBITDA value for element 0, and loop through the rest of the elements of list
    for i in range(1, len(ebitda_list)):
        logs.append(np.log(ebitda_list[i] / ebitda_list[i-1]))

    sigma_l = {}

    # Loop through all the log returns in the 'logs' list, calculate the std of log returns and add to new list.
    for n in range(len(logs)):
        sigma_l.update({gvkey:np.std(logs)})

    return sigma_l



########################################################################################################################
# v_bar functions - small_omega big_omega_g, big_omega_h, g, h, psi_g, psi_h
########################################################################################################################

def small_omega(r, sigma, vstar):
    """
    Adding 0.5 sigma squared to the log normal adjusted drift and then subtracting the risk free interest rate

    :param r: risk free interest rate
    :param sigma: sigma value from sigma() func.
    :param vstar: v_star value from v_star() func.
    :return: value for small letter omega
    """

    return vstar + 0.5 * sigma ** 2 - r


def var_pi(r):
    #In a model with jump risk this would be equal to -(r + lambda_bar), as we ignore jump risk this is simply "-r"
    return -r

def d(a, c):
    #Simplifying function to reduce code of omega/g/h/psi funcs by placing the sqrt portion in its own variable
    return np.sqrt(c ** 2 - 2 * a)


def big_omega_g_plus(a, c):
    return -(d(a, c) - c) / (2 * d(a, c))


def big_omega_g_minus(a, c):
  return +(d(a, c) + c) / (2 * d(a, c))


def big_omega_h_plus(a, c):
    return -(d(a, c) + c) / (2 * d(a, c))


def big_omega_h_minus(a, c):
    return +(d(a, c) - c) / (2 * d(a, c))


def psi_g_plus(a, c):
    return - c - (d(a, c))


def psi_g_minus(a, c):
    return + c - (d(a, c))


def psi_h_plus(a, c):
    return - c + (d(a, c))


def psi_h_minus(a, c):
    return - c + (d(a, c))


def g_plus(a, b, c, y):
    return np.exp(- b * psi_g_plus(a, c) * sp.norm.cdf(((- b - y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def g_minus(a, b, c, y):
    return np.exp(+ b * psi_g_minus(a, c) * sp.norm.cdf(((+ b - y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def h_plus(a, b, c, y):
    return np.exp(- b * psi_h_plus(a, c) * sp.norm.cdf(((- b + y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def h_minus(a, b, c, y):
    return np.exp(+ b * psi_h_minus(a, c) * sp.norm.cdf(((+ b + y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def rho(sigma, vstar, r, mbar, miudelta, couponrate, smallomega, smalla, q):
    # Formula 3.52 - payout_0 formula where we replace A with v_bar
    deriv_payout_0 = ((miu_big_a(r, mbar, sigma) - miudelta) / smallomega) * (big_omega_h_minus(smallomega,
                (vstar + sigma ** 2) / sigma) * (1 - (1 / sigma) * psi_h_minus(smallomega,
                (vstar + sigma ** 2) / sigma)) + big_omega_h_minus(smallomega, (vstar + sigma ** 2) / sigma) *
                (- 2 * smalla - 1 - (1 / sigma) * psi_h_minus(smallomega, (vstar + sigma ** 2) / sigma)) - 1)

    # Formula 3.53 - coupon_0 formula where we replace A with v_bar and isolate v_bar
    deriv_coupon_0 = (couponrate / var_pi(r)) * \
               (- (1 / sigma) * big_omega_h_minus(var_pi(r), (vstar / sigma))
                * psi_h_minus(var_pi(r), (vstar / sigma)) + big_omega_h_minus(var_pi(r), (vstar / sigma)) * (
                    - 2 * smalla - (1 / sigma) * psi_h_minus(var_pi(r), (vstar / sigma))))

    # Formula 3.54 - capex_0 formula where we replace A with v_bar and isolate v_bar
    deriv_capex_0 = (q / var_pi(r)) * (- (1 / sigma) * big_omega_h_minus(var_pi(r), (vstar / sigma)) *
                                 psi_h_minus(var_pi(r), (vstar / sigma)) +
                                 big_omega_h_minus(var_pi(r), (vstar / sigma)) *
                                 (- 2 * smalla - (1 / sigma * psi_h_minus(var_pi(r), (vstar / sigma)))))

    return (deriv_coupon_0 + deriv_capex_0) / deriv_payout_0


def v_bar(rho, bigl):
    """
    If the firm value/asset value passes is lower than this point, the firm defaults.

    :param rho: combination of payout, coupon and capex formulas
    :param bigl: firm liabilities
    :return: Point of default for the firm.
    """
    return rho * bigl


def big_f(a, b, c, y):
    """
    Proposition 8, formula 2.67
    """

    if b > 0:
        bigf = big_omega_g_plus(a,c) * g_plus(a, b, c, y)  + big_omega_h_plus(a, c) * h_plus(a, b, c, y)
    else:
        bigf = big_omega_g_minus(a,c) * g_minus(a, b, c, y)  + big_omega_h_minus(a, c) * h_minus(a, b, c, y)
    return bigf

def payout_0(delta0, r, vstar, sigma, bigr, bigr2a2):
    """
    Formula 3.10
    """
    s_omega = small_omega(r, sigma, vstar)
    aux_omg_h_min_pos = big_omega_h_minus(s_omega,((vstar + sigma ** 2)/sigma))
    aux_omg_h_min_min = big_omega_h_minus(s_omega, - ((vstar + sigma ** 2)/sigma))
    aux_bigr_1 = bigr ** ((1/sigma) * psi_h_minus(s_omega, ((vstar + sigma ** 2) / sigma)))
    aux_bigr_2 = bigr2a2 + ((1 / sigma) * psi_h_minus(s_omega, - ((vstar + sigma ** 2) / sigma)))

    return ((delta0 /s_omega) * (aux_omg_h_min_pos * aux_bigr_1 + aux_omg_h_min_min * aux_bigr_2 - 1))

def coupon_0(couponrate, varpi, v, sigma, bigr, bigr2a):
    """
    Formula 3.15
    """
    aux_big_omg_h_min_pos = big_omega_h_minus(varpi, (v / sigma))
    aux_big_omg_h_min_min = big_omega_h_minus(varpi, - (v / sigma))
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(varpi, (v / sigma)))
    aux_bigr_2 = bigr2a + ((1 / sigma) * psi_h_minus(varpi, - (v / sigma)))

    return ((couponrate / varpi) * (aux_big_omg_h_min_pos * aux_bigr_1 + aux_big_omg_h_min_min * aux_bigr_2 - 1))

def capex_0(q, varpi, v, sigma, bigr, bigr2a):
    """
    Formula 3.16
    """
    aux_big_omg_h_min_pos = big_omega_h_minus(varpi, (v / sigma))
    aux_big_omg_h_min_min = big_omega_h_minus(varpi, - (v / sigma))
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(varpi, (v / sigma)))
    aux_bigr_2 = bigr2a + ((1 / sigma) * psi_h_minus(varpi, - (v / sigma)))

    return ((q / varpi) * (aux_big_omg_h_min_pos * aux_bigr_1 + aux_big_omg_h_min_min * aux_bigr_2 - 1))

########################################################################################################################
# Drift related functions - v , v_star
########################################################################################################################

def small_v(sigma, mbar, miudelta):
    """
    Represents the drift of the process.
    :param sigma:
    :param miudelta: The instantaneous growth rate of the project cash flows (exogeniuously determined).
    :param mbar: The premium per unit of volatility risk

    :return: the drift process
    """

    return miudelta - (mbar * sigma)


def v_star(sigma, smallv):
    """
    Lognormal adjusted drift.

    :return: the drift adjusted for lognormality.
    """

    return smallv - 0.5 * sigma ** 2


def big_l(gvkey, liabilities):
    """
    Retrieve liabilities at time t.

    :param t: time
    :return: value of liabilities at time t
    """

    # Loops through all gvkeys and adds them into the MySQL query


    return


########################################################################################################################
# ... functions
########################################################################################################################

def small_a(vstar, sigma):
    """
    Lognormal adjusted drift divided by sigma squared.

    :return: the value of a based on v_star and sigma.
    """

    return vstar / sigma ** 2


def big_r(vbar, biga0):
    """
    Ratio of the barrier to the current project value.
    Basically has to be < 1 or the firm is already closed.

    :param vbar: barrier value
    :param biga0: value of the security at time 0
    :return: returns the distance from the barrier
    """
    # not sure what this represents..

    return vbar / biga0


def big_r_2a(bigr, smalla):
    """
    Exponent of distance to the barrier.

    :param bigr: distance from the barrier
    :param smalla: lognormal adjusted drift divided by sigma squared
    :return: exponent of the distance to the barrier
    """

    return bigr ** (2 * smalla)


def big_r_2a_2(bigr, smalla):
    """
    Exponent of distance to the barrier, adding two

    :param bigr: distance from the barrier
    :param smalla: lognormal adjusted drift divided by sigma squared
    :return: exponent of the distance to the barrier
    """

    return bigr ** ((2 * smalla) + 2)


########################################################################################################################
# Standard normal distribution functions - h1, h2, h3, h4
########################################################################################################################

def h1(biga0, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s

    :return: value for h1 to be used in standard normal distribution and standard normal density

    """

    # A probably represents the 'asset value', is this equal to big_a_0 or something else?

    return (np.log(z / biga0) - vstar * s) / (sigma * np.sqrt(s))


def h2(bigr, vbar, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h1 to be used in standard normal distribution and standard normal density

    """

    return (np.log(bigr * (vbar / z)) + vstar * s) / (sigma * np.sqrt(s))


def h3(biga0, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h3 to be used in standard normal distribution and standard normal density
    """

    return (np.log(z / biga0) - (vstar + sigma ** 2) * s) / (sigma * np.sqrt(s))


def h4(bigr, vbar, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h4 to be used in standard normal distribution and standard normal density
    """

    return (np.log((bigr * (vbar / z))) + (vstar + sigma ** 2) * s) / (sigma * np.sqrt(s))
