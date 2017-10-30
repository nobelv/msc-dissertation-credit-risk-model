# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

import pandas as pd
import numpy as np
import scipy.stats as sp
import mysql.connector

########################################################################################################################
# State variable functions - A_0, miu_A, sigma, small_r and miu_delta
########################################################################################################################

def big_a_0(delta_0, miu_big_a,):
    """
    The value of a security A_t at the beginning of the process is given by the function A0 = delta_0 / (miu_A - g).
    Where g = miu_delta - (lambda*jump), with jump being equal to 0 g = miu_delta.

    :param delta_0: the value of EBITDA;
    :param miu_A: the discount rate, assumed to be constant for mathematical tractability;
    """

    assets = delta_0 / (miu_big_a - miu_delta())

    return assets


def miu_big_a(r, m_bar):
    """
    The discount rate, assumed to be constant

    :return: the discount rate
    """

    # do we hardcode a discount rate or how do we calculate it?
    # Email said to take parameter as arg but also to code the function?

    discount = r + m_bar * sigma()

    return discount


def miu_delta():
    """
    Instantaneous growth rate of the firm. Essentially the historical growth rate of the state variable,
    which is EBITDA in our model. We query the data from the MySQL database and create a list of growth rates per firm.

    :return: List of lists containing the historical growth rate of each of the firms based on their EBITDA.
    """

    # Connect to database and create cursor object
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost', database='msc_data')

    # Create two cursors to execute commands, one is buffered to allow fetching entire result set for each loop
    cursor = cnx.cursor()
    cursor_buff = cnx.cursor(buffered=True)

    # Query for unique values of gvkey
    q_gvkey = """ SELECT gvkey FROM na_data GROUP BY gvkey """
    cursor.execute(q_gvkey)

    # Use list comprehension to append all unique values of gvkey in database to list
    keys = [gvkey[0] for gvkey in cursor]

    # Query for EBITDA values grouped by gvkey
    # Create empty list to append lists filled with EBITDA data from database
    l = []

    # Loops through the total number of gvkeys and adds them into the MySQL query
    for i in range(len(keys)):
        k = keys[i]
        e_gvkey = ("SELECT ebitda FROM na_data WHERE ebitda > '0' AND gvkey = (%s)")
        cursor_buff.execute(e_gvkey, (k,))

        # Retrieve EBITDA values for each company and add to a list
        ebitda = [float(ebitda[0]) for ebitda in cursor_buff]

        # Check if data is complete, 11 data points, and only if complete add to the list.
        if len(ebitda) >= 11:
            l.append(ebitda)

    # Create empty list to store growth rates
    growth = []

    # Loop over all lists in gvkey list 'l'
    for enterprise in l:

        # Add an empty list to 'l' for each iteration
        growth.append([])

        # Skip element 1, no previous EBITDA value for element 0, and loop through the rest of the elements of list
        for i in range(1, len(enterprise)):
            growth[-1].append((enterprise[i] - enterprise[i - 1])/ enterprise[i - 1])

    # Create empty list to store the average growth of each company
    avg_g = []

    # Loop through all the growth rates in the 'growth' list, calculate the average of the growth rates
    # and add to new list.
    for n in range(len(growth)):
        avg_g.append([np.average(growth[n])])

    # Close connection and cursors
    cnx.close()
    cursor.close()
    cursor_buff.close()

    # Return the list of average EBITDA growth rates

    return avg_g


def small_r():
    """"""

def sigma():
    """
    Queries gvkey data from MySQL database to find the unique values of gvkey and adds them to a list,
    queries EBITDA data based on unique values in gvkey list, calculates lognormal return on EBITDA
    and then the standard deviation.

    :return: List of lists containing the sigma estimate based on lognormal return of EBITDA per company
    """

    # Connect to database and create cursor object
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost', database='msc_data')

    # Create two cursors to execute commands, one is buffered to allow fetching entire result set for each loop
    cursor = cnx.cursor()
    cursor_buff = cnx.cursor(buffered=True)

    # Query for unique values of gvkey
    q_gvkey = """ SELECT gvkey FROM na_data GROUP BY gvkey """
    cursor.execute(q_gvkey)

    # Use list comprehension to append all unique values of gvkey in database to list
    keys = [gvkey[0] for gvkey in cursor]

    # Query for EBITDA values grouped by gvkey
    # Create empty list to append lists filled with EBITDA data from database
    l = []

    # Loops through the total number of gvkeys and adds them into the MySQL query
    for i in range(len(keys)):
        k = keys[i]
        e_gvkey = ("SELECT ebitda FROM na_data WHERE ebitda > '0' AND gvkey = (%s)")
        cursor_buff.execute(e_gvkey, (k,))

        # Retrieve EBITDA values for each company and add to a list
        ebitda = [float(ebitda[0]) for ebitda in cursor_buff]

        # Check if data is complete, 11 data points, and only if complete add to the list.
        if len(ebitda) >= 11:
            l.append(ebitda)

    # Create empty list to store log returns
    logs = []

    # Loop over all lists in gvkey list 'l'
    for enterprise in l:

        # Add an empty list to 'l' for each iteration
        logs.append([])

        # Skip element 1, no previous EBITDA value for element 0, and loop through the rest of the elements of list
        for i in range(1, len(enterprise)):
            logs[-1].append(np.log(enterprise[i] / enterprise[i - 1]))

    # Create empty list to store std of each company
    sigma_l = []

    # Loop through all the log returns in the 'logs' list, calculate the std of log returns and add to new list.
    for n in range(len(logs)):
        sigma_l.append([np.std(logs[n])])

    # Close connection and cursors
    cnx.close()
    cursor.close()
    cursor_buff.close()

    # Return the list of standard deviations
    return sigma_l


########################################################################################################################
# v_bar functions - small_omega big_omega_g, big_omega_h, g, h, psi_g, psi_h
########################################################################################################################
def small_omega(r):
    """
    Adding 0.5 sigma squared to the log normal adjusted drift and then subtracting the risk free interest rate

    :param r: risk free interest rate
    :return: value for small letter omega
    """
    ######################################## have to adjust this to a loop through the sigma() list!
    omg = v_star() + 0.5 * sigma() ** 2 - r
    ######################################## have to adjust this to a loop through the sigma() list!

    return omg


def var_pi(r):
    """
    In a model with jump risk this would be equal to -(r + lambda_bar), as we ignore jump risk this is simply -r

    :param r: risk free interest rate
    :return: negative value for r
    """
    negative_r = -r

    return negative_r


def d(a, c):
    """
    Simplifying function to reduce code of omega/g/h/psi funcs by placing the sqrt portion in its own variable

    """
    squareroot = np.sqrt(c**2 - 2*a)

    return squareroot


def big_omega_g_plus(a, c):
    """"""
    omega_g_plus = -(d(a,c) -c) / (2*d(a,c))

    return omega_g_plus


def big_omega_g_minus(a, c):
    """"""
    omega_g_minus = +(d(a, c) + c) / (2 * d(a, c))

    return omega_g_minus


def big_omega_h_plus(a, c):
    """"""
    omega_h_plus = -(d(a,c) + c) / (2 * d(a,c))

    return omega_h_plus


def big_omega_h_minus(a, c):
    """"""
    omega_h_minus = +(d(a, c) - c) / (2 * d(a, c))

    return omega_h_minus


def psi_g_plus(a, c):
    """"""
    p_g_plus = -c -(d(a,c))

    return p_g_plus


def psi_g_minus(a, c):
    """"""

    p_g_minus = +c -(d(a,c))

    return p_g_minus


def psi_h_plus(a, c):
    """"""
    p_h_plus = - c + (d(a,c))

    return p_h_plus


def psi_h_minus(a, c):
    """"""
    p_h_minus =  - c + (d(a,c))

    return p_h_minus


def g_plus(a, b, c, y):
    """"""
    gplus = np.exp(- b * psi_g_plus(a,c) * sp.norm.cdf(((- b - y * d(a,c)) / np.sqrt(y)) ,loc=0.0, scale=1.0))

    return gplus


def g_minus(a, b, c, y):
    """"""
    gminus = np.exp( + b * psi_g_minus(a,c) * sp.norm.cdf(((+ b - y * d(a,c)) / np.sqrt(y)), loc=0.0, scale=1.0))

    return gminus


def h_plus(a, b, c, y):
    """"""
    hplus = np.exp(- b * psi_h_plus(a,c) * sp.norm.cdf(((- b + y * d(a,c)) / np.sqrt(y)), loc=0.0, scale=1.0))

    return hplus


def h_minus(a, b, c, y):
    """"""
    hminus = np.exp(+ b * psi_h_minus(a,c) * sp.norm.cdf(((+ b + y * d(a,c)) / np.sqrt(y)), loc=0.0, scale=1.0))

    return hminus


def rho(r, m_bar, c):
    """"""

    # Formula 3.52 - payout_0 formula where we replace A with v_bar
    payout_0 = ((miu_big_a(r, m_bar)- miu_delta())/small_omega(r)) * (big_omega_h_minus(small_omega(r),
            (v_star()+ sigma() ** 2) / sigma()) * (1 - (1 / sigma()) *
            psi_h_minus(small_omega(r), (v_star() + sigma() ** 2) / sigma())) +
            big_omega_h_minus(small_omega(r), (v_star()+ sigma() ** 2) / sigma()) *
            ( - 2 * small_a() -1 - (1 / sigma()) *
            psi_h_minus(small_omega(r), (v_star()+ sigma() ** 2) / sigma())) - 1)

    # Formula 3.53 - coupon_0 formula where we replace A with v_bar and isolate v_bar
    # what and where do I find "c"
    coupon_0 = ((c * big_l(0))/var_pi(r)) * (- (1 / sigma()) * big_omega_h_minus(var_pi(r), (v_star() / sigma())) *
                psi_h_minus(var_pi(r), (v_star() / sigma())) + big_omega_h_minus(var_pi(r), (v_star() / sigma())) * (
                - 2 * small_a() - (1 / sigma()) * psi_h_minus(var_pi(r), (v_star() / sigma()))))

    # Formula 3.54 - capex_0 formula where we replace A with v_bar and isolate v_bar
    # what and where do I find "q"
    capex_0 = (q / var_pi(r)) * (- (1 / sigma()) * big_omega_h_minus(var_pi(r), (v_star() / sigma())) *
                psi_h_minus(var_pi(r), (v_star() / sigma())) +
                big_omega_h_minus(var_pi(r), (v_star() / sigma())) *
                (- 2 * small_a() - (1 / sigma()) * psi_h_minus(var_pi(r), (v_star() / sigma()))))

    rho_output = (coupon_0 + capex_0) / payout_0

    return rho_output

def v_bar(rho, big_l):
    """
    If the firm value/asset value passes is lower than this point, the firm defaults.

    :param rho:
    :return: Point of default for the firm.
    """

    vbar = rho * big_l

    return vbar


########################################################################################################################
# Drift related functions - v , v_star
########################################################################################################################

def small_v(m_bar):
    """
    Represents the drift of the process.

    :param miu_delta: The instantaneous growth rate of the project cash flows (exogeniuously determined).
    :param m_bar: The premium per unit of volatility risk

    :return: the drift process
    """

    v = miu_delta() - (m_bar * sigma())

    return v


def v_star(m_bar):
    """
    Lognormal adjusted drift.

    :return: the drift adjusted for lognormality.
    """

    vstar = small_v(m_bar) - 0.5 * sigma()**2

    return vstar


def big_l(t):
    """
    Retrieve liabilities at time t.

    :return:
    """

    cnx = mysql.connector.connect(user='root', password='',
                                  host ='localhost', database ='msc_data')


    l =

    return l


########################################################################################################################
# ... functions
########################################################################################################################

def small_a():
    """
    Lognormal adjusted drift divided by sigma squared.

    :return: the value of a based on v_star and sigma.
    """
    a = v_star()/sigma()**2

    return a


def big_r():
    """
    Distance from the barrier (v_bar).

    :return:
    """
    # not sure what this represents..

    R = v_bar() / big_a_0()

    return R


def big_r_2a():
    """
    Exponent of distance to the barrier.
    :return:
    """
    r2a = big_r()**(2*small_a())

    return r2a


def big_r_2a_2():
    """
    Exponent of distance to the barrier.

    :return:
    """
    r2a2 = big_r()**((2*small_a())+2)

    return r2a2


########################################################################################################################
# Standard normal distribution functions - h1, h2, h3, h4
########################################################################################################################

def h1(z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h1 to be used in standard normal distribution and standard normal density

    """

    # A probably represents the 'asset value', is this equal to big_a_0 or something else?

    output_h1 = (np.log(z / big_a_0()) - v_star() * s) / (sigma() * np.sqrt(s))


    return output_h1


def h2(z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h1 to be used in standard normal distribution and standard normal density

    """
    output_h2 = (np.log(big_r() * (v_bar() / z)) + v_star() * s) / (sigma() * np.sqrt(s))

    return output_h2


def h3(z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h3 to be used in standard normal distribution and standard normal density
    """

    output_h3 = (np.log(z / big_a_0()) - (v_star() + sigma() ** 2) * s) / (sigma() * np.sqrt(s))

    return output_h3


def h4(z, s):
    """
    Standard normal distribution inputs.

    :param z: input specified by security formula
    :param s: time denoted as s
    :return: value for h4 to be used in standard normal distribution and standard normal density
    """

    output_h4 = (np.log((big_r() * (v_bar() / z))) + (v_star() + sigma() ** 2) * s) / (sigma() * np.sqrt(s))

    return output_h4