from lifelines import KaplanMeierFitter
import autograd.numpy as np
from autograd.scipy.stats import norm
import math
from lifelines.utils.safe_exp import safe_exp
from lifelines.fitters import ParametricUnivariateFitter
from lifelines.fitters import ParametricRegressionFitter
import matplotlib.pyplot as plt
import sys

result = {}

f = open('data.txt',"r")
precipitation=[]
for line in f:
    val = float(line.strip('\n'))
    val = val if val > 0 else sys.float_info.min
    precipitation.append(val)

fig, ax = plt.subplots()
date_range = range(1960, 2016)

ax.set(title='Total monthly precipitation during June in São Carlos, Brazil')
plt.xlabel('year', fontsize=10)
plt.ylabel('precipitation (mm)', fontsize=15)
ax.scatter(date_range, precipitation, s=12)
ax.grid()
plt.show()

def cleanVal(val, places=6):
    return str(round(float(val), places))

def clean(val):
     return val if val != "" else "-"

def cleanStandards(fitter, key):
    try:
        se = fitter._compute_standard_errors()[key].to_list()[0]
        count = 56
        return cleanVal(se) #+ " / " + cleanVal(se * np.sqrt(count))
    except:
        return " - "
    
class EP(ParametricUnivariateFitter):
    _fitted_parameter_names = ['lambda_', 'beta_']  
    _bounds = ((0, None), (None, None))
    _KNOWN_MODEL = True
    def _cumulative_hazard(self, params, times):
        lambda_, beta_ = params
        CDF = (safe_exp(lambda_ * safe_exp(-1 * beta_ * times)) - safe_exp(lambda_))/(1 - safe_exp(lambda_)).cumsum()
        v = -np.log( 1 - CDF )
        return v

EP_Fitter = EP()
EP_Fitter.fit(precipitation, event_observed=date_range)

result["EP"] = { 
    "lambda" : cleanVal(EP_Fitter.params_["lambda_"]) + " / " + cleanStandards(EP_Fitter, "lambda_"),
    "beta" : cleanVal(EP_Fitter.params_["beta_"]) + " / " + cleanStandards(EP_Fitter, "beta_"),
    "p" : "",
    "AIC" : EP_Fitter.AIC_,
    "BIC" : EP_Fitter.BIC_,
}

print(result["EP"])

class Zero_I_EP(ParametricUnivariateFitter):
    _fitted_parameter_names = ['lambda_', 'beta_']
    _bounds = ((0, None), (None, None))
    _KNOWN_MODEL = True
    def _cumulative_hazard(self, params, times):
        lambda_, beta_ = params
        CDF = safe_exp(-1 * lambda_) + 1 - safe_exp(-1* lambda_ + lambda_ * safe_exp(-1 * beta_ * times))
        v = -1 * np.log( 1 - CDF )
        return v

Zero_I_EP_Fitter = Zero_I_EP()
Zero_I_EP_Fitter.fit(precipitation, event_observed=date_range)

result["ZI EP"] = { 
    "lambda" : cleanVal(Zero_I_EP_Fitter.params_["lambda_"]) + " / " + cleanStandards(Zero_I_EP_Fitter, "lambda_"),
    "beta" : cleanVal(Zero_I_EP_Fitter.params_["beta_"]) + " / " + cleanStandards(Zero_I_EP_Fitter, "beta_"),
    "p" : "",
    "AIC" : Zero_I_EP_Fitter.AIC_,
    "BIC" : Zero_I_EP_Fitter.BIC_
}

print(result["ZI EP"])

class Hurdle_EP(ParametricUnivariateFitter):
    _fitted_parameter_names = ['p_' , 'lambda_', 'beta_']
    _bounds = ((0, 1), (None, None), (None, None))
    _KNOWN_MODEL = True
    def _cumulative_hazard(self, params, times):
        p_, lambda_, beta_ = params
        f_l_b = (safe_exp(lambda_ * safe_exp(-1 * beta_ * times)) - safe_exp(lambda_))/(1 - safe_exp(lambda_))
        CDF = (p_ + ((1 - p_) * f_l_b))
        v = -1 * np.log( 1 - CDF )
        return v

Hurdle_EP_Fitter = Hurdle_EP()
Hurdle_EP_Fitter.fit(precipitation, event_observed=date_range)

result["Hurdle EP"] = { 
    "lambda" : cleanVal(Hurdle_EP_Fitter.params_["lambda_"]) + " / " + cleanStandards(Hurdle_EP_Fitter, "lambda_"),
    "beta" : cleanVal(Hurdle_EP_Fitter.params_["beta_"]) + " / " + cleanStandards(Hurdle_EP_Fitter, "beta_"),
    "p" : cleanVal(Hurdle_EP_Fitter.params_["p_"], 10) + " / " + cleanStandards(Hurdle_EP_Fitter, "p_"),
    "AIC" : Hurdle_EP_Fitter.AIC_,
    "BIC" : Hurdle_EP_Fitter.BIC_,
}

print(result["Hurdle EP"])

class Zero_I_Exp(ParametricUnivariateFitter):
    _fitted_parameter_names = ['p_', 'beta_']
    _bounds = ((0, 1), (None, None))
    _KNOWN_MODEL = True
    def _cumulative_hazard(self, params, times):
        p_, beta_ = params
        CDF = 1 - ( ( 1 - p_ ) * math.e ** ( -1 * beta_ * times ) )
        v = -1 * np.log( 1 - CDF )
        return v

Zero_I_Exp_Fitter = Zero_I_Exp()
Zero_I_Exp_Fitter.fit(precipitation, event_observed=date_range)

result["ZI EXP"] = { 
    "lambda" : "",
    "beta" : cleanVal(Zero_I_Exp_Fitter.params_["beta_"]) + " / " + cleanStandards(Zero_I_Exp_Fitter, "beta_"),
    "p" : cleanVal(Zero_I_Exp_Fitter.params_["p_"], 10) + " / " + cleanStandards(Zero_I_Exp_Fitter, "p_"),
    "AIC" : Zero_I_Exp_Fitter.AIC_,
    "BIC" : Zero_I_Exp_Fitter.BIC_
}

print(result["ZI EXP"])

KMF_Fitter = KaplanMeierFitter()
KMF_Fitter.fit(precipitation, event_observed=date_range)

ax = EP_Fitter.plot_survival_function(ci_show=False)
ax = Zero_I_EP_Fitter.plot_survival_function(ci_show=False)
ax = Hurdle_EP_Fitter.plot_survival_function(ci_show=False)
ax = Zero_I_Exp_Fitter.plot_survival_function(ci_show=False)
ax = KMF_Fitter.plot_survival_function(ax=ax, ci_show=False)
plt.show()

print("-"*120)
print("{:<20} {:<20}".format("", "Note : Parameters in form : Value / Standard Error"))
print("-"*120)
print("{:<20} {:<40} {:<40} {:<40}".format("Model", "λ", "β", "p"))
print("-"*120)
for k, v in result.items():
    lambda_, beta_, p_, aic_, bic_ = v
    l = clean(v[lambda_])
    b = clean(v[beta_])
    p = clean(v[p_])
    aic = clean(v[aic_])
    bic = clean(v[bic_])
    print ("{:<20} {:<40} {:<40} {:<40}".format(k,l,b,p))
print("-"*120)
print()
print()
print("{:<20} {:<40} {:<40}".format("Model", "AIC", "BIC"))
print("-"*120)
for k, v in result.items():
    lambda_, beta_, p_, aic_, bic_ = v
    aic = cleanVal(v[aic_], 5)
    bic = cleanVal(v[bic_], 5)
    print ("{:<20} {:<40} {:<40}".format(k,aic, bic))
print("-"*120)

