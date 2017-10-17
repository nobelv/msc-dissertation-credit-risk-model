# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

import pandas as pd
import numpy as np
import mysql.connector

########################################################################################################################
# State variable functions - A_0, miu_A, sigma
########################################################################################################################

def big_a_0(delta_0, miu_big_a, g):
    """
    The value of a security A_t at the beginning of the process is given by this function.

    :param delta_0: the value of EBITDA;
    :param miu_A: the discount rate, assumed to be constant for mathematical tractability;
    :param g: the expected CFO growth rate, g = miu_delta - lambda * jump, but we ignore jumps so g is simply miu_delta.
    """
    # g can't be both an input and calculated in the formula but not sure which is right

    assets = delta_0 / (miu_big_a - g)


    return assets


def miu_big_a(r, m_bar, sigma):
    """
    The discount rate, assumed to be constant

    :return: the discount rate
    """

    # do we hardcode a discount rate or how do we calculate it?
    # Email said to take parameter as arg but also to code the function?

    discount = r + m_bar * sigma

    return discount


def sigma():
    """
    Queries data from mySQL database to find the unique values of gvkey and adds to list,
    Queries EBITDA data based on unique values, calculates lognormal return on EBITDA and then the standard deviation.

    :return: The sigma estimate based on lognormal return of EBITDA
    """

    # Connect to database and create cursor object
    cnx = mysql.connector.connect(user='root', password='',
                                  host='127.0.0.1', database='msc_data')

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

    # Loop goes through the total number of gvkeys and adds them into the SQL query
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

    # Return the list of standard deviations
    return sigma_l


########################################################################################################################
# Drift related functions - v , v_star, v_bar
########################################################################################################################

def small_v(miu_delta, m_bar, sigma):
    """
    Represents the drift of the process.

    :param miu_delta: The instantaneous growth rate of the project cash flows (exogeniuously determined).
    :param m_bar: The premium per unit of volatility risk

    :return: the drift process
    """

    v = miu_delta - (m_bar * sigma)

    return v


def v_star():
    """
    Lognormal adjusted drift.

    :return: the drift adjusted for lognormality.
    """

    vstar = small_v() - 0.5 * sigma()**2

    return vstar


def big_l(t):
    """
    Retrieve liabilities at time t.

    :return:
    """

    cnx = mysql.connector.connect(user='root', password='',
                                  host ='127.0.0.1', database ='msc_data')


    l =

    return l



def v_bar(rho, big_l):
    """
    If the firm value/asset value passes is lower than this point, the firm defaults.

    :param rho:
    :return: Point of default for the firm.
    """



    vbar = rho * big_l

    return vbar

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

    :param z: input specified by the securities
    :param s: time s
    :return: value for h1 to be used in standard normal distribution and standard normal density

    """

    # A probably represents the 'asset value', is this equal to big_a_0 or something else?

    output_h1 = (np.log(z / big_a_0()) - v_star() * s) / (sigma() * np.sqrt(s))


    return output_h1


def h2(z, s):
    """

     :param z:
     :param s:
     :return: value for h2 to be used in standard normal distribution and standard normal density

     """
    output_h2 = (np.log(big_r()*(v_bar()/z)) + v_star()* s) / (sigma() * np.sqrt(s))

    return output_h2


def h3(z, s):
    """

     :param z:
     :param s:
     :return:

     """

    return


def h4(z, s):
    """

     :param z:
     :param s:
     :return:

     """

    return