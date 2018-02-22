# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Model Execution file, requires model_functions.py & data_functions.py!
import model_functions as mf
import scipy.optimize as sp
import pandas as pd
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
df = pd.read_csv(path + "\Data\cashflow_based_statevar.csv", sep=",",
                 dtype={"key": str, "fyear": str, "ticker": str, "statevariable": float, "fixedcosts": float,
                        "couponrate": float, "liabilities": float, "equityobserved": float})

df['fyear'] = pd.to_datetime(df['fyear'], format="%m%d%Y").apply(lambda x: x.date())

df2 = pd.read_csv(path + "\Data\\na_treasury.csv", sep=';')
df2['thedate'] = pd.to_datetime(df2['thedate'], format="%Y%m%d")
df2 = df2.set_index(['thedate'])

keys = df.ticker.unique()
statevar_dict = {}
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
    liabilities = dataframe['liabilities']
    fixedcosts = dataframe['fixedcosts']
    fyear = dataframe['fyear']
    date = dataframe['fyear']
    equityobs = dataframe['equityobserved']

    statevar_dict.update({k: statevar})
    liabilities_dict.update({k: liabilities})
    fixedcosts_dict.update({k: fixedcosts})
    fyear_dict.update({k: fyear})
    date_dict.update({k: date})
    equityobs_dict.update({k: equityobs})

########################################################################################################################
# Executing the model
########################################################################################################################
miu_delta_dict = {}
sigma_dict = {}
mbar_list = []
sigma_list = []
miudelta_list = []

for i in range(len(keys)):
    k = keys[i]
    miudelta, sigma = mf.sigma_and_miu(k, statevar_dict)
    sigma_dict.update({k: sigma})
    miu_delta_dict.update({k: miudelta})

    for t in range(len(statevar_dict[k])):
        statevar_list = statevar_dict[k].tolist()
        fixedcosts_list = fixedcosts_dict[k].tolist()
        delta_0 = statevar_list[t]
        liabilities_list = liabilities_dict[k].tolist()
        equityobserved = equityobs_dict[k].tolist()
        couponrate = df['couponrate'].tolist()

        fyear = fyear_dict[k].tolist()
        dates = date_dict[k].tolist()
        rf_rate = df2.iloc[df2.index.get_loc(dates[t], method="nearest")].tolist()

        # Set the initial 'guess' for the x0 to be used in Newton-Raphson algo
        x0 = ((miu_delta_dict[k] - rf_rate[0]) / sigma_dict[k]) + 0.001

        def quadratic(x):
            return (mf.effective_taxrate("usa") *
                    (mf.payout_0(delta_0, rf_rate[0],
                                 mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k],
                                 mf.big_r(mf.v_bar(sigma_dict[k], mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                   rf_rate[0], x, miu_delta_dict[k], couponrate[t], liabilities_list[t],
                                                   mf.small_omega(rf_rate[0], sigma_dict[k],
                                                                  mf.v_star(miu_delta_dict[k], x, sigma_dict[k])),
                                                   mf.small_a(mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                              sigma_dict[k]), fixedcosts_list[t]),
                                          mf.big_a_0(delta_0, mf.miu_big_a(rf_rate[0], x, sigma_dict[k]),
                                                     miu_delta_dict[k])), mf.small_a(
                            mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k])) -

                     mf.coupon_0(couponrate[t], liabilities_list[t], mf.var_pi(rf_rate[0]),
                                 mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k],
                                 mf.big_r(mf.v_bar(sigma_dict[k], mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                   rf_rate[0], x, miu_delta_dict[k], couponrate[t], liabilities_list[t],
                                                   mf.small_omega(rf_rate[0], sigma_dict[k],
                                                                  mf.v_star(miu_delta_dict[k], x, sigma_dict[k])),
                                                   mf.small_a(mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                              sigma_dict[k]), fixedcosts_list[t]),
                                          mf.big_a_0(delta_0, mf.miu_big_a(rf_rate[0], x, sigma_dict[k]),
                                                     miu_delta_dict[k])), mf.small_a(
                             mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k])) -

                     mf.capex_0(fixedcosts_list[t], mf.var_pi(rf_rate[0]), mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                sigma_dict[k],
                                mf.big_r(mf.v_bar(sigma_dict[k], mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                  rf_rate[0], x, miu_delta_dict[k], couponrate[t], liabilities_list[t],
                                                  mf.small_omega(rf_rate[0], sigma_dict[k],
                                                                 mf.v_star(miu_delta_dict[k], x, sigma_dict[k])),
                                                  mf.small_a(mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                             sigma_dict[k]), fixedcosts_list[t]),
                                         mf.big_a_0(delta_0, mf.miu_big_a(rf_rate[0], x, sigma_dict[k]),
                                                    miu_delta_dict[k])), mf.small_a(
                             mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k])))) - equityobserved[t]
        try:
            mbar = sp.newton(quadratic, x0)
            mbar_list.append(mbar)
            miudelta_list.append(miudelta)
            sigma_list.append(sigma)
            # print function can be enabled for debugging.
            # print("Risk Premium for company", k, "is", mbar, "for date", fyear[t])
        except RuntimeError:
            print("Failed to converge after 50 iterations.", "Attempted calculation for company", k,
                  "for date", fyear[t])
            pass

mbar_list = pd.Series(mbar_list)
sigma_list = pd.Series(sigma_list)
miudelta_list = pd.Series(miudelta_list)

df['mbar'] = mbar_list.values
df['sigma'] = sigma_list.values
df['miu_delta'] = miudelta_list.values

print(df.head())

writer = pd.ExcelWriter(path + "\Data\model_outputs.xlsx")
df.to_excel(writer, 'Company Data - M_Bar')
writer.save()
writer.close()

########################################################################################################################
# Stopping the timer
elapsed = timeit.default_timer() - start_time
print("Code executed in", '{:5.2f}'.format(elapsed), "seconds.")
########################################################################################################################
