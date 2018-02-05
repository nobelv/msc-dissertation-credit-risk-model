# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

import numpy as np
import scipy.stats as sp
import statsmodels.api as sm

########################################################################################################################
# State variable functions - A_0, miu_A, sigma, small_r and miu_delta
########################################################################################################################


def big_a_0(delta0, miua, miudelta):
    """
    The value of a security A_t at the beginning of the process is given by the function A0 = delta_0 / (miu_A - g).
    Where g = miu_delta - (lambda*jump), with jump being equal to 0 g = miu_delta.

    :param delta0: the value of EBITDA at t = 0.
    :param miua: the discount rate, assumed to be constant for mathematical tractability (miu_big_a).
    :param miudelta: instantaneous growth rate of the firm.

    :return: The value of the security at time 0.
    """

    return delta0 / (miua - miudelta)


def miu_big_a(r, mbar, sigma):
    """
    The discount rate, assumed to be constant

    :return: the discount rate
    """

    return float(r) + mbar * sigma


def miu_delta(gvkey, ebitdadict, fyeardict, sigma):
    """
    Instantaneous growth rate of the firm. Essentially the historical growth rate of the state variable,
    which is EBITDA in our model. We use data from the MySQL database and estimate a growth rate through
    a robust linear regression. For this model we've decided to use HuberT's robust regression.

    :param gvkey: gvkey corresponding to the firm.
    :param ebitdadict: dictionary containing all gvkeys and their ebitda values.
    :param fyeardict: dictionary containing all gvkeys and the years of which we have data.
    :param sigma: standard deviation of the lognormal return on EBITDA.

    :return: List of lists containing the historical growth rate of each of the firms based on their EBITDA.
    """

    y = np.array(ebitdadict[gvkey].tolist())
    x = np.array(fyeardict[gvkey].tolist())

    x = np.reshape(x, (x.shape[0], -1))
    x = sm.add_constant(x)

    ln_y = np.log(y)

    rlm_model = sm.RLM(ln_y, x, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    beta_1 = rlm_results.params[1]

    miudelta = beta_1 + 0.5 * sigma ** 2

    return miudelta


def func_sigma(gvkey, ebitdadict):
    """
    Lognormal return on EBITDA and the standard deviation of these returns.

    :param gvkey: The gvkey corresponding to the firm.
    :param ebitdadict: The dictionary containing all gvkeys and their ebitda values.

    :return: dictionary containing the sigma estimate based on lognormal return of EBITDA per company.
    """
    ebitda_list = ebitdadict[gvkey].tolist()
    logs = []

    # Skip element 1, no previous EBITDA value for element 0, and loop through the rest of the elements of list
    for i in range(1, len(ebitda_list)):
        logs.append(np.log(ebitda_list[i] / ebitda_list[i-1]))

    sigma_dict = {}

    # Loop through all the log returns in the 'logs' list, calculate the std of log returns and add to new dictionary.
    for n in range(len(logs)):
        sigma_dict.update({gvkey: np.std(logs)})

    return sigma_dict


########################################################################################################################
# v_bar functions - small_omega big_omega_g, big_omega_h, g, h, psi_g, psi_h
########################################################################################################################


def small_omega(r, sigma, vstar):
    """
    Adding 0.5 sigma squared to the log normal adjusted drift and then subtracting the risk free interest rate

    :param r: The risk free interest rate.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param vstar: The lognormal adjusted drift.

    :return: The log normal adjusted drift plus 0.5 sigma squared minus the risk free interest rate.
    """

    return vstar + 0.5 * sigma ** 2 - r


def var_pi(r):
    """
    In a model with jump risk this would be equal to -(r + lambda_bar), However, our model ignores jump risk.
    As such var_pi is simply the risk free interest rate with a swapped sign.

    :param r: The risk free interest rate.

    :return: The swapped sign risk free interest rate.
    """
    return -r


# Simplifying function to reduce code of omega/g/h/psi funcs by placing the sqrt portion in its own variable
def d(a, c):
    return np.sqrt(c ** 2 - 2 * a)


# Auxiliary functions found on page 33
def big_omega_g_plus(a, c):
    return - (d(a, c) - c) / (2 * d(a, c))


def big_omega_g_minus(a, c):
    return (d(a, c) + c) / (2 * d(a, c))


def big_omega_h_plus(a, c):
    return - (d(a, c) + c) / (2 * d(a, c))


def big_omega_h_minus(a, c):
    return (d(a, c) - c) / (2 * d(a, c))


def psi_g_plus(a, c):
    return - c - (d(a, c))


def psi_g_minus(a, c):
    return c - (d(a, c))


def psi_h_plus(a, c):
    return - c + (d(a, c))


def psi_h_minus(a, c):
    return c + (d(a, c))


def g_plus(a, b, c, y):
    return np.exp(- b * psi_g_plus(a, c) * sp.norm.cdf(((- b - y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def g_minus(a, b, c, y):
    return np.exp(+ b * psi_g_minus(a, c) * sp.norm.cdf(((+ b - y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def h_plus(a, b, c, y):
    return np.exp(- b * psi_h_plus(a, c) * sp.norm.cdf(((- b + y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def h_minus(a, b, c, y):
    return np.exp(+ b * psi_h_minus(a, c) * sp.norm.cdf(((+ b + y * d(a, c)) / np.sqrt(y)), loc=0.0, scale=1.0))


def v_bar(sigma, vstar, r, mbar, miudelta, couponrate, liabilities, smallomega, smalla, q):
    """
    The Default barrier. If the firm value/asset value passes is lower than this point, the shareholders
    give up the firm and the firm defaults.

    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param vstar: The lognormal adjusted drift.
    :param r: The risk free interest rate.
    :param mbar: The market price of risk.
    :param miudelta: the instantaneous growth rate of the firm.
    :param couponrate: The firm's interest expense coupon rate.
    :param liabilities: The firm's total debt.
    :param smallomega: The log normal adjusted drift plus 0.5 sigma squared minus the risk free interest rate.
    :param smalla: Lognormal adjusted drift divided by sigma squared.
    :param q: The firm's nominal capital expenditure.

    :return: The level of the default barrier.
    """
    # Formula 3.52 - payout_0 formula where we replace A with v_bar
    a1 = smallomega
    c1 = (vstar + sigma ** 2) / sigma
    deriv_payout_0 = ((miu_big_a(r, mbar, sigma) - miudelta) / smallomega) * \
                     (big_omega_h_minus(a1, c1) * (1 - (1 / sigma) * psi_h_minus(a1, c1)) +
                      big_omega_h_minus(a1, - c1) *
                      (- 2 * smalla - 1 - (1 / sigma) * psi_h_minus(a1, - c1)) - 1)

    # Formula 3.53 - coupon_0 formula where we replace A with v_bar and isolate v_bar
    a2 = var_pi(r)
    c2 = (vstar / sigma)
    deriv_coupon_0 = ((couponrate * liabilities) / var_pi(r)) * \
                     (- (1 / sigma) * big_omega_h_minus(a2, c2) * psi_h_minus(a2, c2) + big_omega_h_minus(a2, - c2) *
                      (- 2 * smalla - (1 / sigma) * psi_h_minus(a2, - c2)))

    # Formula 3.54 - capex_0 formula where we replace A with v_bar and isolate v_bar
    deriv_capex_0 = (q / a2) * \
                    (- (1 / sigma) * big_omega_h_minus(a2, c2) *
                     psi_h_minus(a2, c2) + big_omega_h_minus(a2, - c2) *
                     (- 2 * smalla - (1 / sigma * psi_h_minus(a2, - c2))))

    return (deriv_coupon_0 + deriv_capex_0) / deriv_payout_0


def big_f(a, b, c, y):
    """
    Proposition 8, formula 2.67.
    """

    if b > 0:
        bigf = big_omega_g_plus(a, c) * g_plus(a, b, c, y) + big_omega_h_plus(a, c) * h_plus(a, b, c, y)
    else:
        bigf = big_omega_g_minus(a, c) * g_minus(a, b, c, y) + big_omega_h_minus(a, c) * h_minus(a, b, c, y)
    return bigf


def payout_0(delta0, r, vstar, sigma, bigr, smalla):
    """
    Formula 3.10
    Discounted sum of all future cash flows as long as the firm exists.

    :param delta0: The value of EBITDA at t = 0.
    :param r: The risk free interest rate.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.

    :return:
    """
    s_omega = small_omega(r, sigma, vstar)
    a = s_omega
    c = (vstar + sigma ** 2) / sigma
    aux_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** ((2 * smalla) + 2 + (1 / sigma) * psi_h_minus(a, - c))

    return (delta0 / s_omega) * ((aux_omg_h_min_pos * aux_bigr_1) + (aux_omg_h_min_min * aux_bigr_2 - 1))


def coupon_0(couponrate, liabilities, varpi, vstar, sigma, bigr, smalla):
    """
    Formula 3.15
    Discounted sum of all future interest costs as long as the firm exists.

    :param couponrate: The firm's interest expense coupon rate.
    :param liabilities: The firm's total debt.
    :param varpi: The swapped sign risk free interest rate.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.

    :return:
    """
    a = varpi
    c = (vstar / sigma)
    aux_big_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_big_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** (2 * smalla + (1 / sigma) * psi_h_minus(a, - c))

    return ((couponrate * liabilities) / varpi) * \
           ((aux_big_omg_h_min_pos * aux_bigr_1) + (aux_big_omg_h_min_min * aux_bigr_2) - 1)


def capex_0(q, varpi, vstar, sigma, bigr, smalla):
    """
    Formula 3.16
    Discounted sum of all future capex costs as long as the firm exists.

    :param q: The firm's nominal capital expenditure.
    :param varpi: The swapped sign risk free interest rate.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param bigr: Ratio of the barrier to the current project value.
    :param smalla: Lognormal adjusted drift divided by sigma squared.

    :return:
    """
    a = varpi
    c = (vstar / sigma)
    aux_big_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_big_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** (2 * smalla + (1 / sigma) * psi_h_minus(a, - c))

    return (q / varpi) * ((aux_big_omg_h_min_pos * aux_bigr_1) + (aux_big_omg_h_min_min * aux_bigr_2) - 1)


def effective_taxrate(taxcorp, taxdiv):
    return (1-taxcorp)*(1-taxdiv)


def div0(effectivetax, payout, coupon, capex):
    return (1 - effectivetax) * (payout - coupon - capex)


########################################################################################################################
# Drift related functions - v , v_star
########################################################################################################################
def rho(vbar, liabilities):
    """
    A scaling factor, the barrier (v_bar) as normalized by the firm's debt.

    :param vbar: The barrier as calculated by derivatives of payout, coupon and capex functions.
    :param liabilities: The firm's debt.

    :return: A scaling factor, the barrier as normalized by the firm's debt.
    """
    return vbar / liabilities


def v_star(miudelta, mbar, sigma):
    """
    Lognormal adjusted drift of the process.

    :param miudelta: The instantaneous growth rate of the project cash flows (exogeniuously determined).
    :param mbar: The premium per unit of volatility risk.
    :param sigma: The standard deviation of the lognormal return on EBITDA.

    :return: The drift adjusted for lognormality.
    """

    return miudelta - (mbar * sigma) - 0.5 * (sigma ** 2)


########################################################################################################################
# ... functions
########################################################################################################################

def small_a(vstar, sigma):
    """
    Lognormal adjusted drift divided by sigma squared.

    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.

    :return: the value of a based on v_star and sigma.
    """

    return vstar / sigma ** 2


def big_r(vbar, biga0):
    """
    Ratio of the barrier to the current project value.
    Basically has to be < 1 or the firm is already closed.

    :param vbar: The barrier value.
    :param biga0: The value of the security at time 0.

    :return: returns the distance from the barrier.
    """

    return vbar / biga0


########################################################################################################################
# Standard normal distribution functions - h1, h2, h3, h4
########################################################################################################################

def h1(biga0, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param biga0: The value of the security at time 0.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param z: input specified by security formula.
    :param s: time denoted as s.

    :return: value for h1 to be used in standard normal distribution and standard normal density.

    """

    return (np.log(z / biga0) - vstar * s) / (sigma * np.sqrt(s))


def h2(bigr, vbar, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param bigr: Ratio of the barrier to the current project value.
    :param vstar: The lognormal adjusted drift.
    :param vbar: The barrier as calculated by derivatives of payout, coupon and capex functions.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param z: input specified by security formula.
    :param s: time denoted as s.

    :return: value for h1 to be used in standard normal distribution and standard normal density.

    """

    return (np.log(bigr * (vbar / z)) + vstar * s) / (sigma * np.sqrt(s))


def h3(biga0, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param biga0: The value of the security at time 0.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param z: input specified by security formula.
    :param s: time denoted as s.

    :return: value for h3 to be used in standard normal distribution and standard normal density.
    """

    return (np.log(z / biga0) - (vstar + sigma ** 2) * s) / (sigma * np.sqrt(s))


def h4(bigr, vbar, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param bigr: Ratio of the barrier to the current project value.
    :param vbar: The barrier as calculated by derivatives of payout, coupon and capex functions.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard deviation of the lognormal return on EBITDA.
    :param z: input specified by security formula.
    :param s: time denoted as s.

    :return: value for h4 to be used in standard normal distribution and standard normal density.
    """

    return (np.log(bigr * (vbar / z)) + (vstar + sigma ** 2) * s) / (sigma * np.sqrt(s))
