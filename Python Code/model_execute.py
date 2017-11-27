# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Model Execution file, requires model_functions.py & securities_functions.py!

import numpy as np
import scipy.stats as sp
import mysql.connector
import model_functions as mf
import securities_functions as sf

cnx = mysql.connector.connect(user='root', password='', host='localhost', database='msc_data')

# Create two cursors to execute commands, one is buffered to allow fetching entire result set for each loop
cursor = cnx.cursor()
cursor_buff = cnx.cursor(buffered=True)

query_gvkey = "SELECT gvkey FROM na_market_data GROUP BY gvkey"
cursor.execute(query_gvkey)
keys = [gvkey[0] for gvkey in cursor]

ebitda_dict = {}
for i in range(len(keys)):
    k = keys[i]
    query_ebitda = "SELECT ebitda FROM na_market_data WHERE ebitda > '0' AND gvkey = (%s)"
    cursor_buff.execute(query_ebitda, (k,))
    ebitda = [float(ebitda[0]) for ebitda in cursor_buff]
    ebitda_dict.update({k: ebitda})

liabilities_dict = {}
for i in range(len(keys)):
    k = keys[i]
    query_liabilities = "SELECT lt FROM na_market_data WHERE gvkey = (%s) LIMIT 1"
    cursor_buff.execute(query_liabilities, (k,))
    liabilities = [float(liabilities[0]) for liabilities in cursor_buff]
    liabilities_dict.update({k: liabilities})

coupon_dict = {}
for i in range(len(keys)):
    k = keys[i]
    query_coupon = "SELECT couponrate FROM na_market_data WHERE gvkey = (%s) LIMIT 1"
    cursor_buff.execute(query_coupon, (k,))
    coupons = [float(coupons[0]) for coupons in cursor_buff]
    coupon_dict.update({k: coupons})

capex_dict = {}
for i in range(len(keys)):
    k = keys[i]
    query_capex = "SELECT capx FROM na_market_data WHERE gvkey = (%s) LIMIT 1"
    cursor_buff.execute(query_capex, (k,))
    capex = [float(capex[0]) for capex in cursor_buff]
    capex_dict.update({k: capex})

cnx.close()
cursor.close()
cursor_buff.close()
