# Dissertation Credit Risk Modeling
# Indicator of the market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Model Execution file, requires model_functions.py & data_functions.py!
import model_functions_nodebt as mf
import statsmodels.api as sm
import scipy.optimize as sp
import scipy.stats as sps
import pandas as pd
import numpy as np
import timeit
import os

########################################################################################################################
# Starting a timer
start_time = timeit.default_timer()
########################################################################################################################

########################################################################################################################
# Setting up dictionaries with all the data
########################################################################################################################
path = os.path.dirname(os.path.abspath(""))

# Reading the data into a dataframe from our csv files.
df = pd.read_csv(path + "\Data\cashflow_data.csv", sep=",",
                 dtype={"fyear": str, "ticker": str, "firmname": str, "industry": str, "sector": str,
                        "statevariable": float, "cash": float, "fixedcosts": float, "couponrate": float,
                        "liabilities": float, "equityobserved": float})

df['fyear'] = pd.to_datetime(df['fyear'], format="%m%d%Y").apply(lambda x: x.date())

df2 = pd.read_csv(path + "\Data\\na_treasury.csv", sep=';')
df2['thedate'] = pd.to_datetime(df2['thedate'], format="%Y%m%d")
df2 = df2.set_index(['thedate'])

print('Data succesfully loaded into dataframe.')

keys = df.ticker.unique()
print("Dataset contains", len(keys), "firms.")
statevar_dict = {}
cash_dict = {}
liabilities_dict = {}
coupon_dict = {}
fixedcosts_dict = {}
fyear_dict = {}
date_dict = {}
equityobs_dict = {}

# Populating our dictionaries with data specific to each company.
for key in range(len(keys)):
    k = keys[key]
    dataframe = df.loc[df['ticker'] == k]
    statevar = dataframe['statevariable']
    cash = dataframe['cash']
    liabilities = dataframe['liabilities']
    fixedcosts = dataframe['fixedcosts']
    fyear = dataframe['fyear']
    date = dataframe['fyear']
    equityobs = dataframe['equityobserved']

    statevar_dict.update({k: statevar})
    cash_dict.update({k: cash})
    liabilities_dict.update({k: liabilities})
    fixedcosts_dict.update({k: fixedcosts})
    fyear_dict.update({k: fyear})
    date_dict.update({k: date})
    equityobs_dict.update({k: equityobs})

print('Successfully populated dictionaries. Beginning model execution.')

########################################################################################################################
# Executing the model
########################################################################################################################
miu_delta_dict = {}
sigma_dict = {}
mbar_list = []
sigma_list = []
miudelta_list = []
miu_a_list = []
vbar_list = []


# Lists for statistical tests
sw_list = []
corr_list = []

mean_rev_counter = 0
correl_counter = 0
sw_counter = 0
fail_counter = 0

for i in range(len(keys)):
    k = keys[i]
    # test for correlation between equity and statevar
    correlation = np.corrcoef(equityobs_dict[k], statevar_dict[k])
    correlation = correlation[1].item(0)

    if correlation > 0:
        # if correlation positive do mean reversion test

        statevar = np.asarray(statevar_dict[k])
        y = (statevar[1:] - statevar[:-1]) / statevar[:-1]
        X = 1 / statevar[:-1]
        X = sm.add_constant(X)
        rlm_model = sm.RLM(y, X, M=sm.robust.norms.LeastSquares())
        rlm_results = rlm_model.fit()

        if rlm_results.params[0] < 0 and rlm_results.pvalues[0] < 0.05:
            mean_rev_counter += 1
            # print function can be enabled for debugging.
            # print("Firm:", k, "fails the mean reversion test.", rlm_results.params[0], rlm_results.pvalues[0])
            df = df[~df['ticker'].isin([k])]
        else:
            *_, sw = sps.shapiro(statevar_dict[k])
            if sw > 0.05:
                for n in range(len(statevar_dict[k])):
                    corr_list.append(correlation)
                    sw_list.append(sw)

                miudelta, sigma = mf.sigma_and_miu(k, statevar_dict)
                sigma_dict.update({k: sigma})
                miu_delta_dict.update({k: miudelta})

                for t in range(len(statevar_dict[k])):
                    statevar_list = statevar_dict[k].tolist()
                    cash_list = cash_dict[k].tolist()
                    fixedcosts_list = fixedcosts_dict[k].tolist()
                    delta_0 = statevar_list[t]
                    cash_0 = cash_list[t]
                    liabilities_list = liabilities_dict[k].tolist()
                    equityobserved = equityobs_dict[k].tolist()
                    couponrate = df['couponrate'].tolist()

                    fyear = fyear_dict[k].tolist()
                    dates = date_dict[k].tolist()
                    rf_rate = df2.iloc[df2.index.get_loc(dates[t], method="nearest")].tolist()

                    # Create shorthands used in the function quadratic(m_bar)
                    sigm = sigma_dict[k]
                    r = rf_rate[0]
                    miu_delta = miu_delta_dict[k]
                    c = couponrate[t]
                    L = liabilities_list[t]
                    q = fixedcosts_list[t]
                    eq_obs = equityobserved[t]
                    varpi = mf.var_pi(r)

                    # Set the initial 'guess' for the x0 to be used in root finding algo
                    x0 = ((miu_delta - r) / sigm) + 0.001

                    def quadratic(m_bar):

                        divtax = mf.div_taxrate()

                        payout_0 = mf.payout_0(delta_0, r, mf.v_star(miu_delta, m_bar, sigm), sigm,
                                             mf.big_r(mf.v_bar(sigm, mf.v_star(miu_delta, m_bar, sigm),
                                                               r, m_bar, miu_delta, c, L,
                                             mf.omega(r, sigm, mf.v_star(miu_delta, m_bar, sigm)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm), q),
                                             mf.big_a_0(delta_0, mf.miu_big_a(r, m_bar, sigm), miu_delta)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm))

                        coupon_0 = mf.coupon_0(c, L, varpi,
                                             mf.v_star(miu_delta, m_bar, sigm), sigm,
                                             mf.big_r(mf.v_bar(sigm, mf.v_star(miu_delta, m_bar, sigm),
                                                               r, m_bar, miu_delta, c, L,
                                             mf.omega(r, sigm, mf.v_star(miu_delta, m_bar, sigm)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm), q),
                                             mf.big_a_0(delta_0, mf.miu_big_a(r, m_bar, sigm), miu_delta)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm))

                        fixedcost_0 = mf.fixedcost_0(q, varpi, mf.v_star(miu_delta, m_bar, sigm), sigm,
                                             mf.big_r(mf.v_bar(sigm, mf.v_star(miu_delta, m_bar, sigm),
                                                                  r, m_bar, miu_delta, c, L,
                                             mf.omega(r, sigm, mf.v_star(miu_delta, m_bar, sigm)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm), q),
                                             mf.big_a_0(delta_0, mf.miu_big_a(r, m_bar, sigm), miu_delta)),
                                             mf.small_a(mf.v_star(miu_delta, m_bar, sigm), sigm))

                        # print("payout", payout_0)
                        # print("coupon", coupon_0)
                        # print("fixed costs", fixedcost_0)
                        # print("equity", eq_obs)
                        # print("cash", cash_0)

                        return divtax * (cash_0 + payout_0 - coupon_0 - fixedcost_0) - eq_obs

                    try:
                        mbar = sp.newton(quadratic, x0)
                        mbar_list.append(mbar)
                        miudelta_list.append(miudelta)
                        sigma_list.append(sigm)
                        miua = mf.miu_big_a(rf_rate[0], mbar, sigm)
                        miu_a_list.append(miua)

                        # calculate barrier
                        vstar = mf.v_star(miudelta, mbar, sigm)
                        omega = mf.omega(r, sigm, vstar)
                        smalla = mf.small_a(vstar, sigm)

                        vbar = mf.v_bar(sigm, vstar, r, mbar, miudelta, c, L, omega, smalla, q)
                        vbar_list.append(vbar)

                    except RuntimeError:
                        # print("Failed to converge after 50 iterations.", "Attempted calculation for company", k,
                        # "for date", fyear[t])
                        fail_counter += 1
                        mbar = 99.99
                        mbar_list.append(mbar)
                        miudelta_list.append(miudelta)
                        sigma_list.append(sigm)
                        miua = mf.miu_big_a(rf_rate[0], mbar, sigm)
                        miu_a_list.append(miua)
                        vbar_list.append(99.99)
                        pass
            else:
                sw_counter += 1
                df = df[~df['ticker'].isin([k])]

    else:
        correl_counter += 1
        # print("Firm:", k, "fails the correlation test.", correlation) # for debug
        df = df[~df['ticker'].isin([k])]

print('Completed running the model, writing data to output file.')

keys_count = df.ticker.unique()
print("Model succesfully ran for", len(keys_count), "firms. Filtering out", len(keys) - len(keys_count), "firms.")
print(correl_counter, "firms filtered due to correlation,", mean_rev_counter, "firms filtered due to mean reversion.",
      sw_counter, "firms filtered due to SW test.")
print("fails", fail_counter)

mbar_list = pd.Series(mbar_list)
sigma_list = pd.Series(sigma_list)
miudelta_list = pd.Series(miudelta_list)
miu_a_list = pd.Series(miu_a_list)
sw_list = pd.Series(sw_list)
corr_list = pd.Series(corr_list)
vbar_list = pd.Series(vbar_list)

df['mbar'] = mbar_list.values
df['sigma'] = sigma_list.values
df['miu_delta'] = miudelta_list.values
df['miu_A'] = miu_a_list.values
df['Shapiro-Wilk'] = sw_list.values
df['Correlation'] = corr_list.values
df['vbar'] = vbar_list.values

writer = pd.ExcelWriter(path + "\Model Output\model_output_nodebt_test2.xlsx")
df.to_excel(writer, sheet_name="Model Output")
writer.save()
writer.close()

########################################################################################################################
# Stopping the timer
elapsed = timeit.default_timer() - start_time
print("Model executed in", '{:5.2f}'.format(elapsed), "seconds.")
########################################################################################################################
