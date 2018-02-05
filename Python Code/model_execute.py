# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Model Execution file, requires model_functions.py & securities_functions.py!
import model_functions as mf
import newtonraphson as nr
import pandas as pd
from operator import itemgetter
import timeit

########################################################################################################################
# Starting a timer
start_time = timeit.default_timer()
########################################################################################################################

########################################################################################################################
# Setting up dictionaries with all the data
########################################################################################################################

# Reading the data into a dataframe from our csv files.
df = pd.read_csv("g:/na_data.csv", sep=';')
df2 = pd.read_csv("g:/na_treasury.csv", sep=';')
df2['thedate'] = pd.to_datetime(df2['thedate'])
df2 = df2.set_index(['thedate'])

keys = df.gvkey.unique()
ebitda_dict = {}
liabilities_dict = {}
coupon_dict = {}
capex_dict = {}
fyear_dict = {}

# Populating our dictionaries with data specific to each company.
for i in range(len(keys)):
    k = keys[i]
    data = df.loc[df['gvkey'] == k]
    ebitda = data['ebitda']
    liabilities = data['liabilities']
    couponrate = data['couponrate']
    capex = data['capex']
    fyear = data['fyear']
    ebitda_dict.update({k: ebitda})
    liabilities_dict.update({k: liabilities})
    coupon_dict.update({k: couponrate})
    capex_dict.update({k: capex})
    fyear_dict.update({k: fyear})


########################################################################################################################
# Executing the model
########################################################################################################################

# Obtain sigma and miu_delta
miu_delta_dict = {}
sigma_dict = {}
for i in range(len(keys)):
    k = keys[i]
    # Populate dictionaries with sigma & miu_delta values
    sigma = mf.func_sigma(k, ebitda_dict)
    sigma_dict.update({k: sigma[k]})
    miudelta = mf.miu_delta(k, ebitda_dict, fyear_dict, sigma[k])
    miu_delta_dict.update({k: miudelta})

# Obtain risk free rate and calculate mbar
for i in range(len(keys)):
    k = keys[i]
    # get the first value in ebitda and fyear per company
    delta_0 = itemgetter(0)(ebitda_dict[k].tolist())
    year0 = str(itemgetter(0)(fyear_dict[k].tolist()))

    start_date = "{}-12-01".format(year0)
    end_date = "{}-12-31".format(year0)

    temp = df2.loc[start_date:end_date]
    rf_rate = temp['tenyr_rate'].tail(1)
    rf_rate = rf_rate.tolist()

    def quadratic(x):
        return float(rf_rate[0]) + float(x) * sigma_dict[k]

    mbar = nr.solve(quadratic, 1, 0.01)
    miua = mf.miu_big_a(rf_rate[0], mbar, sigma_dict[k])
    biga0 = mf.big_a_0(delta_0, miua, miu_delta_dict[k])
    print(rf_rate[0])
    print(sigma_dict[k])
    print(mbar)
    print(miua)
    print(biga0)

########################################################################################################################
# Stopping the timer
elapsed = timeit.default_timer() - start_time
print("Code executed in", '{:5.2f}'.format(elapsed), "seconds.")
########################################################################################################################
