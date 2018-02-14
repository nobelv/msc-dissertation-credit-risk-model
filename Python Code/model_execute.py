# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Model Execution file, requires model_functions.py & securities_functions.py!
import model_functions as mf
import data_functions as dfunc
import scipy.optimize as sp
import pandas as pd
import timeit

########################################################################################################################
# Starting a timer
start_time = timeit.default_timer()
########################################################################################################################

########################################################################################################################
# Setting up dictionaries with all the data
########################################################################################################################

# Reading the data into a dataframe from our csv files.
df = pd.read_csv("g:/na_quarterly_1990.csv", sep=';', dtype={"gvkey": int, "cusip": str, "conml": str, "datadate": str,
                                                             "fyearq": int, "fqtr": int, "niq": float, "ltq": float,
                                                             "seqq": float, "capxytd": float, "tieq": float,
                                                             "xintq": float, "intexp": float, "txtq": float,
                                                             "dpq": float, "ebitda": float, "prccq": float,
                                                             "cshoq": float, "equityobs": float})
df2 = pd.read_csv("g:/na_treasury2.csv", sep=';')
df2['thedate'] = pd.to_datetime(df2['thedate'], format="%Y%m%d")
df2 = df2.set_index(['thedate'])

keys = df.gvkey.unique()
ebitda_dict = {}
liabilities_dict = {}
coupon_dict = {}
capex_dict = {}
intexp_dict = {}
fyear_dict = {}
fqtr_dict = {}
date_dict = {}
equityobs_dict = {}

# Populating our dictionaries with data specific to each company.
for key in range(len(keys)):
    k = keys[key]
    dataframe = df.loc[df['gvkey'] == k]
    qtr = dataframe['fqtr']
    ebitda = dataframe['ebitda']
    intexp = dataframe['intexp']

    ebitda_dict.update({k: ebitda})
    intexp_dict.update({k: intexp})
    fqtr_dict.update({k: qtr})

    liabilities = dataframe['ltq']
    capex = dataframe['capxytd']
    fyear = dataframe['fyearq']

    liabilities_dict.update({k: liabilities})
    capex_dict.update({k: capex})
    fyear_dict.update({k: fyear})

    date = dataframe['datadate']
    date_dict.update({k: date})

    equityobs = dataframe['equityobs']
    equityobs_dict.update({k: equityobs})

# Annualize ebitda, interest expense and capex

ebitda_ann = dfunc.annualize_quarterly(keys, ebitda_dict, fqtr_dict)
intexp_ann = dfunc.annualize_quarterly(keys, intexp_dict, fqtr_dict)
capex_ann = dfunc.annualize_qytd(keys, capex_dict, fqtr_dict)

########################################################################################################################
# Executing the model
########################################################################################################################

miu_delta_dict = {}
sigma_dict = {}

for i in range(len(keys)):
    k = keys[i]
    # Populate dictionaries with sigma & miu_delta values
    sigma = mf.func_sigma(k, ebitda_ann)
    sigma_dict.update({k: sigma[k]})
    miudelta = mf.miu_delta(k, ebitda_ann, fyear_dict, sigma[k])
    miu_delta_dict.update({k: miudelta})

    x0 = 0.3

    for t in range(len(ebitda_ann[k])):
        ebitdalist = ebitda_ann[k]
        capex_list = capex_ann[k]
        intexp_list = intexp_ann[k]
        delta_0 = ebitdalist[t]

        dates = date_dict[k].tolist()
        rf_rate = df2.iloc[df2.index.get_loc(dates[t], method="nearest")].tolist()

        liabilities_list = liabilities_dict[k].tolist()

        coupondata_dict = {'liabilities': liabilities_list, 'interest_exp': intexp_list}
        coupon_df = pd.DataFrame(coupondata_dict)
        coupon_df['coupon_rate'] = coupon_df['interest_exp'] / coupon_df['liabilities']
        couponrate = coupon_df['coupon_rate'].tolist()

        equityobserved = equityobs_dict[k].tolist()

        fqtr = fqtr_dict[k].tolist()
        fyear = fyear_dict[k].tolist()

        def quadratic(x):
            return (mf.effective_taxrate("usa") *
                    (mf.payout_0(delta_0, rf_rate[0],
                                 mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k],
                                 mf.big_r(mf.v_bar(sigma_dict[k], mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                   rf_rate[0], x, miu_delta_dict[k], couponrate[t], liabilities_list[t],
                                                   mf.small_omega(rf_rate[0], sigma_dict[k],
                                                                  mf.v_star(miu_delta_dict[k], x, sigma_dict[k])),
                                                   mf.small_a(mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                              sigma_dict[k]), capex_list[t]),
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
                                                              sigma_dict[k]), capex_list[t]),
                                          mf.big_a_0(delta_0, mf.miu_big_a(rf_rate[0], x, sigma_dict[k]),
                                                     miu_delta_dict[k])), mf.small_a(
                             mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k])) -

                     mf.capex_0(capex_list[t], mf.var_pi(rf_rate[0]), mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                sigma_dict[k],
                                mf.big_r(mf.v_bar(sigma_dict[k], mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                  rf_rate[0], x, miu_delta_dict[k], couponrate[t], liabilities_list[t],
                                                  mf.small_omega(rf_rate[0], sigma_dict[k],
                                                                 mf.v_star(miu_delta_dict[k], x, sigma_dict[k])),
                                                  mf.small_a(mf.v_star(miu_delta_dict[k], x, sigma_dict[k]),
                                                             sigma_dict[k]), capex_list[t]),
                                         mf.big_a_0(delta_0, mf.miu_big_a(rf_rate[0], x, sigma_dict[k]),
                                                    miu_delta_dict[k])), mf.small_a(
                             mf.v_star(miu_delta_dict[k], x, sigma_dict[k]), sigma_dict[k])))) - equityobserved[t]
        try:
            mbar = sp.newton(quadratic, x0)
            x0 = mbar
            print("Risk Premium for company", k, "is", mbar, "in fiscal year", fyear[t], "for quarter", fqtr[t])
        except RuntimeError:
            print("Failed to converge after 50 iterations.", "Attempted calculation for company", k,
                  "in fiscal year", fyear[t], "for quarter", fqtr[t])
            pass


########################################################################################################################
# Stopping the timer
elapsed = timeit.default_timer() - start_time
print("Code executed in", '{:5.2f}'.format(elapsed), "seconds.")
########################################################################################################################
