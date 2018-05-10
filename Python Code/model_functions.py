# Dissertation Credit Risk Modeling
# Indicator of the market price of risk implied in stocks
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
    Where g = miu_delta - (lambda*jump), with jump being equal to 0, g = miu_delta.

    :param delta0: the value of the state variable at t = 0.
    :param miua: the discount rate, assumed to be constant for mathematical tractability (miu_big_a).
    :param miudelta: instantaneous growth rate of the firm.

    :return: The value of the security at time 0.
    """

    return delta0 / (miua - miudelta)


def miu_big_a(r, mbar, sigma):
    """
    The discount rate, assumed to be constant

    :param r: The risk free rate.
    :param mbar: The risk premium.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :return: The discount rate.
    """

    return float(r) + mbar * sigma


def sigma_and_miu(gvkey, statevar_dict, fixedmiu=False):
    """
    Calculates the instantaneous growth rate of the firm, the miu_delta, through a robust linear regression on the
    differences between the log of the state variable. The sigma comes from the standard error of the residuals after
    applying the robust weights.

    :param gvkey: The gvkey corresponding to the firm.
    :param statevar_dict: The dictionary containing all gvkeys and the state variable values.
    :param fixedmiu: determines whether to use a fixed value of miu_delta or not.

    :return: Returns a tuple containing miu_delta (instantaneous growth rate of the firm) and sigma (robustly weighted
    standard error of the residuals).
    """
    statevar = np.asarray(statevar_dict[gvkey])
    y = np.log(statevar[1:]) - np.log(statevar[:-1])
    x = np.ones(len(y))

    rlm_model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    # used for debugging
    # print(rlm_results.summary(yname='y',xname=['var_%d' % i for i in range(len(rlm_results.params))]))

    sigma_calc = np.std(rlm_results.resid * rlm_results.weights)
    miudelta = rlm_results.params[0] + (0.5 * sigma_calc ** 2)
    if fixedmiu is True:
        miudelta = 0.04
    return miudelta, sigma_calc

########################################################################################################################
# Functions used in the equity and barrier calculation - omega var_pi alpha
########################################################################################################################


def omega(r, sigma, vstar):
    """
    Adding 0.5 sigma squared to the log normal adjusted drift and then subtracting the risk free interest rate

    :param r: The risk free interest rate.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param vstar: The lognormal adjusted drift.

    :return: The log normal adjusted drift plus 0.5 sigma squared minus the risk free interest rate.
    """

    return vstar + 0.5 * sigma ** 2 - r


def var_pi_tilde(r, alpha):
    """
    In a model with jump risk this would be equal to -(r + lambda_bar), However, our model ignores jump risk.
    As such var_pi is simply the risk free interest rate with a swapped sign.

    :param r: The risk free interest rate.
    :param alpha: The debt growth rate.

    :return: The swapped sign risk free interest rate adjusted for the debt growth rate.
    """
    return - (r - alpha)


def alpha_debt(miudelta):
    """
    The debt growth rate, assumed to be equal to the growth rate of the firm, miu_delta.

    :param miudelta: the instantaneous growth rate of the firm.

    :return: the debt growth rate.
    """

    return miudelta

########################################################################################################################
# Drift related functions - rho, v_star, v_tilde, small_a, small_a_tilde and big_r
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
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.

    :return: The drift adjusted for lognormality.
    """

    return miudelta - (mbar * sigma) - 0.5 * (sigma ** 2)


def v_tilde(vstar, alpha):
    """
    Debt growth rate adjusted drift. Uses the already lognormal adjusted drift.
        
    :param vstar: The lognormal adjusted drift of the process.
    :param alpha: The debt growth rate.
    
    :return: The lognormal adjusted drift minus the debt growth rate. 
    """

    return vstar - alpha


def small_a(vstar, sigma):
    """
    Lognormal adjusted drift divided by sigma squared.

    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.

    :return: the lognormal adjusted drift divided by sigma squared.

    """

    return vstar / sigma ** 2


def a_tilde(vtilde, sigma):
    """
    Debt growth rate adjusted drift divided by sigma squared. Uses the already lognormal adjusted drift.

    :param vtilde: Debt growth rate adjusted drift. Uses the already lognormal adjusted drift.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.

    :return: the debt growth rate
    """

    return vtilde / sigma ** 2


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
# Barrier and Equity functions - v_bar, payout_0, coupon_0, capex_0, effective_taxrate and div0
########################################################################################################################


def v_bar(sigma, vstar, vtilde, r, mbar, miudelta, couponrate, liabilities, omega, smallatilde, q, alpha):
    """
    The Default barrier. If the firm value/asset value passes is lower than this point, the shareholders
    give up the firm and the firm defaults.

    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param vstar: The lognormal adjusted drift.
    :param vtilde: vstar adjusted by the debt growth rate.
    :param r: The risk free interest rate.
    :param mbar: The market price of risk.
    :param miudelta: the instantaneous growth rate of the firm.
    :param couponrate: The firm's interest expense coupon rate.
    :param liabilities: The firm's total debt.
    :param omega: The log normal adjusted drift plus 0.5 sigma squared minus the risk free interest rate.
    :param smallatilde: The debt growth rate adjusted drift divided by sigma squared, uses log normal adjusted drift.
    :param q: The firm's total fixed costs.
    :param alpha: The debt growth rate.

    :return: The level of the default barrier.
    """
    # Formula 3.52 - payout_0 formula where we replace A with v_bar
    a1 = omega
    c1 = (vtilde + sigma ** 2) / sigma
    deriv_payout_0 = ((miu_big_a(r, mbar, sigma) - miudelta) / omega) * \
                     (big_omega_h_minus(a1, c1) * (1 - (1 / sigma) * psi_h_minus(a1, c1)) +
                      big_omega_h_minus(a1, - c1) *
                      (- 2 * smallatilde - 1 - (1 / sigma) * psi_h_minus(a1, - c1)) - 1)

    # Formula 3.53 - coupon_0 formula where we replace A with v_bar and isolate v_bar
    a2 = var_pi_tilde(r, alpha)
    c2 = (vstar / sigma)
    deriv_coupon_0 = ((couponrate * liabilities) / var_pi_tilde(r, alpha)) * \
                     (- (1 / sigma) * big_omega_h_minus(a2, c2) * psi_h_minus(a2, c2) + big_omega_h_minus(a2, - c2) *
                      (- 2 * smallatilde - (1 / sigma) * psi_h_minus(a2, - c2)))

    # Formula 3.54 - fixedcost_0 formula where we replace A with v_bar and isolate v_bar

    deriv_fixedcost_0 = (q / a2) * \
                    (- (1 / sigma) * big_omega_h_minus(a2, c2) *
                     psi_h_minus(a2, c2) + big_omega_h_minus(a2, - c2) *
                     (- 2 * smallatilde - (1 / sigma * psi_h_minus(a2, - c2))))

    # netdebt_0 formula where we replace A with v_bar and isolate v_bar
    c3 = v_tilde(vstar, alpha) / sigma
    deriv_netdebt_0 = ((alpha * liabilities) / var_pi_tilde(r, alpha) *
                       - (1 / sigma) * big_omega_h_minus(a2, c3) * psi_h_minus(a2, c3) + big_omega_h_minus(a2, - c3) *
                       (- 2 * smallatilde - (1 / sigma) * psi_h_minus(a2, - c3)))

    return ((deriv_coupon_0 + deriv_fixedcost_0) - deriv_netdebt_0) / deriv_payout_0


def big_f(a, b, c, y):
    """
    Proposition 8, formula 2.67.
    """

    if b > 0:
        bigf = big_omega_g_plus(a, c) * g_plus(a, b, c, y) + big_omega_h_plus(a, c) * h_plus(a, b, c, y)
    else:
        bigf = big_omega_g_minus(a, c) * g_minus(a, b, c, y) + big_omega_h_minus(a, c) * h_minus(a, b, c, y)
    return bigf


def payout_0(delta0, r, vstar, vtilde, sigma, bigr, atilde):
    """
    Formula 3.10.

    Discounted sum of all future cash flows as long as the firm exists.

    :param delta0: The value of our cash flow based state variable at t = 0.
    :param r: The risk free interest rate.
    :param vstar: The lognormal adjusted drift.
    :param vtilde: vstar adjusted by the debt growth rate.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param bigr: Ratio of the barrier to the current project value.
    :param atilde: The debt growth rate adjusted drift divided by sigma squared, uses log normal adjusted drift.

    :return: The value of the discounted sum of all future cash flows.
    """
    s_omega = omega(r, sigma, vstar)
    a = s_omega
    c = (vtilde + sigma ** 2) / sigma
    aux_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** ((2 * atilde) + 2 + (1 / sigma) * psi_h_minus(a, - c))

    return (delta0 / s_omega) * ((aux_omg_h_min_pos * aux_bigr_1) + (aux_omg_h_min_min * aux_bigr_2) - 1)


def coupon_0(couponrate, liabilities, varpitilde, vstar, sigma, bigr, smallatilde):
    """
    Formula 3.15

    Discounted sum of all future interest costs as long as the firm exists.

    :param couponrate: The firm's interest expense coupon rate.
    :param liabilities: The firm's total debt.
    :param varpitilde: The swapped sign version of the risk free interest rate minus the debt growth rate.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param bigr: Ratio of the barrier to the current project value.
    :param smallatilde: The debt growth rate adjusted drift divided by sigma squared, uses log normal adjusted drift.

    :return: The value of the discounted sum of all future interest costs.
    """
    a = varpitilde
    c = (vstar / sigma)
    aux_big_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_big_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** (2 * smallatilde + (1 / sigma) * psi_h_minus(a, - c))

    return ((couponrate * liabilities) / varpitilde) * \
           ((aux_big_omg_h_min_pos * aux_bigr_1) + (aux_big_omg_h_min_min * aux_bigr_2) - 1)


def fixedcost_0(q, varpitilde, vstar, sigma, bigr, smallatilde):
    """
    Formula 3.16

    Discounted sum of all future fixed costs as long as the firm exists.

    :param q: The firm's total fixed costs.
    :param varpitilde: The swapped sign version of the risk free interest rate minus the debt growth rate.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param bigr: Ratio of the barrier to the current project value.
    :param smallatilde: The debt growth rate adjusted drift divided by sigma squared, uses log normal adjusted drift.

    :return: The value of the discounted sum of all future fixed costs.
    """
    a = varpitilde
    c = (vstar / sigma)
    aux_big_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_big_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** (2 * smallatilde + (1 / sigma) * psi_h_minus(a, - c))

    return (q / varpitilde) * ((aux_big_omg_h_min_pos * aux_bigr_1) + (aux_big_omg_h_min_min * aux_bigr_2) - 1)


def netdebt_0(alpha, liabilities, varpitilde, vtilde, sigma, bigr, smallatilde):
    """
    Discounted sum of all future new debt issued as long as the firm exists.

    :param alpha: The debt growth rate.
    :param liabilities: The firm's debt.
    :param varpitilde: The swapped sign version of the risk free interest rate minus the debt growth rate.
    :param vtilde: vstar adjusted by the debt growth rate.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param bigr: Ratio of the barrier to the current project value.
    :param smallatilde: The debt growth rate adjusted drift divided by sigma squared, uses log normal adjusted drift.

    :return: The value of the discounted sum of all future new debt issued.
    """
    a = varpitilde
    c = (vtilde / sigma)
    aux_big_omg_h_min_pos = big_omega_h_minus(a, c)
    aux_big_omg_h_min_min = big_omega_h_minus(a, - c)
    aux_bigr_1 = bigr ** ((1 / sigma) * psi_h_minus(a, c))
    aux_bigr_2 = bigr ** (2 * smallatilde + (1 / sigma) * psi_h_minus(a, - c))

    return ((alpha * liabilities) / varpitilde) * ((aux_big_omg_h_min_pos * aux_bigr_1) +
                                                   (aux_big_omg_h_min_min * aux_bigr_2) - 1)


def div_taxrate():
    """
    Used to calculate the effective tax rate.

    :return: Returns the percentage effective tax rate as a float.
    """
    taxdiv = 0.20

    return 1 - taxdiv


def div0(effectivetax, cash, payout, coupon, fixedcosts, netdebt):
    """
    Calculate the value of dividends which is equal to equity assuming there is no equity recovery by shareholders..

    :param effectivetax: The effective tax rate for the market.
    :param cash: The value of cash at t0.
    :param payout: The value of the discounted sum of all future cash flows.
    :param coupon: The value of the discounted sum of all future interest costs.
    :param fixedcosts: The value of the discounted sum of all future fixed costs.

    :return: The company's equity value according to the model.
    """
    return (1 - effectivetax) * (cash + payout - coupon - fixedcosts + netdebt)


########################################################################################################################
# Standard normal distribution functions - h1, h2, h3, h4
########################################################################################################################

def h1(biga0, vstar, sigma, z, s):
    """
    Standard normal distribution inputs.

    :param biga0: The value of the security at time 0.
    :param vstar: The lognormal adjusted drift.
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
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
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
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
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
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
    :param sigma: The standard error of the residuals of the robust linear regression on the state variable.
    :param z: input specified by security formula.
    :param s: time denoted as s.

    :return: value for h4 to be used in standard normal distribution and standard normal density.
    """

    return (np.log(bigr * (vbar / z)) + (vstar + sigma ** 2) * s) / (sigma * np.sqrt(s))

########################################################################################################################
# Auxiliary functions - omega & psi
########################################################################################################################


# Simplifying function to reduce code of omega/g/h/psi funcs by placing the sqrt portion in its own variable
def d(a, c):
    return np.sqrt(c ** 2 - 2 * a)


# Auxiliary functions found on page 33 in Silva (2017)
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

