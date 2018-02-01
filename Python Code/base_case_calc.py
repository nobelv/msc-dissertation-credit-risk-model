# Dissertation Credit Risk Modeling
# Market price of risk implied in stocks
# Credit Risk Model functions - ignoring any jump process

# Base case calculations to make sure the model is working correctly.

import model_functions as mf

# base inputs
delta0 = 400
miu_delta = 0.04
sigma = 0.15
liabilities = 1200
q = 100
r = 0.01
mbar = 0.4
tax_effective = 0.4312
couponrate = 0.0211117441989073

# functions

vstar = mf.v_star(miu_delta, mbar, sigma)
miua = mf.miu_big_a(r, mbar, sigma)
biga0 = mf.big_a_0(delta0, miua, miu_delta)
smalla = mf.small_a(vstar, sigma)
omega = mf.small_omega(r, sigma, vstar)

vbar = mf.v_bar(sigma, vstar, r, mbar, miu_delta, couponrate, liabilities, omega, smalla, q)
bigr = mf.big_r(vbar, biga0)
varpi = mf.var_pi(r)

payout = mf.payout_0(delta0, r, vstar, sigma, bigr, smalla)
coupon = mf.coupon_0(couponrate, liabilities, varpi, vstar, sigma, bigr, smalla)
capex = mf.capex_0(q, varpi, vstar, sigma, bigr, smalla)

equity = mf.div0(tax_effective, payout, coupon, capex)

print("omega =", omega)
print("v star =", vstar, "\n", "miu_a =", miua, "\n", "A0 =", biga0, "\n",
      "a =", smalla, "\n", "R =", bigr, "\n", "varpi =", varpi, "")
print("v bar =", vbar)
print("payout =", payout, "\n", "coupon =", coupon, "\n", "capex =", capex)
print("equity =", equity)
