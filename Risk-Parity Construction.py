import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from scipy.optimize import minimize, Bounds
import scipy.stats as stats
import datetime
from typing import Union
from scipy.interpolate import interp1d
from functools import reduce
from visualization_base import correlation_plot, fan_chart, default_colors
import sklearn.mixture as mixture


sdate = datetime.datetime(1960,1,1)   # start date
edate = datetime.datetime(2023,1,1)
num_assets = 4

all_themes = pd.read_csv(filepath_or_buffer = r'C:\Users\mads-\OneDrive\G&B\Speciale\Kode\Data\Test data\[world]_[all_themes]_[daily]_[vw_cap].csv', sep = ',', header = 0, usecols = ['date','ret']); all_themes.rename(columns = {'ret':'all_themes'}, inplace = True)
momentum_return = pd.read_csv(filepath_or_buffer = r'C:\Users\mads-\OneDrive\G&B\Speciale\Kode\Data\Test data\[world]_[momentum]_[daily]_[vw_cap].csv', sep = ',', header = 0, usecols = ['date','ret']); momentum_return.rename(columns = {'ret':'momentum_return'}, inplace = True)
low_risk_return = pd.read_csv(filepath_or_buffer = r'C:\Users\mads-\OneDrive\G&B\Speciale\Kode\Data\Test data\[world]_[low_risk]_[daily]_[vw_cap].csv', sep = ',', header = 0, usecols = ['date','ret']); low_risk_return.rename(columns = {'ret':'low_risk_return'}, inplace = True)
size_return = pd.read_csv(filepath_or_buffer = r'C:\Users\mads-\OneDrive\G&B\Speciale\Kode\Data\Test data\[world]_[size]_[daily]_[vw_cap].csv', sep = ',', header = 0, usecols = ['date','ret']); size_return.rename(columns = {'ret':'size_return'}, inplace = True)
names = ['all_themes','momentum_return','low_risk_return','size_return']

dfs = [all_themes,momentum_return, low_risk_return, size_return]
all_returns = reduce(lambda left,right: pd.merge(left,right,on='date'), dfs)


all_returns['date'] = pd.to_datetime(all_returns['date'])
returns_filtered = all_returns.loc[(all_returns['date'] >= sdate) & (all_returns['date'] < edate)]


dates = returns_filtered['date']
returns_filtered = returns_filtered.set_index('date')
returns = np.array(returns_filtered)

"""""
plt.plot(dates, returns)
plt.legend(names)
plt.show()
"""

#----------------------------------------------------------------------------------- Covariance matrix in normal state

num_periods = len(returns)
window_size = 5 * 365
effective_num_periods = num_periods - window_size
half_life = 365 * 2
time_points = np.arange(1, window_size + 1)

def cov_to_corr_matrix(cov_mat: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    """

    vols = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    return corr_mat


#Calculate weighing scheme for covariance matrix
def calculate_exponential_decay_probabilities(target_time_point: Union[int, float], time_points: np.ndarray,
                                              half_life: Union[float, int]) -> np.ndarray:

    numerator = np.exp(-np.log(2) / half_life * np.clip(target_time_point - time_points, 0, np.inf))
    denominator = np.sum(numerator)

    p_t = numerator / denominator

    return p_t

#Calculating the covariance matrix
def calculate_cov_mat(x: np.ndarray, probs: np.ndarray, axis: int = 0) -> np.ndarray:

    x = x.T if axis == 1 else x

    expected_x_squared = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, x), axis=0)
    mu = probs @ x
    mu_squared = np.einsum('j, i -> ji', mu, mu)
    cov_mat = expected_x_squared - mu_squared

    return cov_mat


exp_probs = calculate_exponential_decay_probabilities(window_size, time_points, half_life)

"""""
plt.plot(exp_probs)
"""""

cov_mat = np.zeros((effective_num_periods,num_assets,num_assets))

for t in range(effective_num_periods):

    cov_mat[t,:,:] = calculate_cov_mat(returns_filtered.iloc[t: window_size + t, :].values, probs = exp_probs)


def check_positive_semi_definite_eig(matrix: np.ndarray):
    """
    Checks that a symmetric square matrix is positive semi-deinite.
    """

    return np.all(np.linalg.eigvals(matrix) >= 0)

"""""
counts = np.zeros(effective_num_periods)
for t in range(effective_num_periods):
    counts[t] = (check_positive_semi_definite_eig(cov_mat[t,:,:]) if False else 0)
print(np.sum(counts))
"""""
"""
print(cov_mat)
plt.plot(dates[365*5:],cov_mat[:,1,0])
plt.show()
"""

avg_cov_mat = np.zeros((num_assets,num_assets))

for i in range(num_assets):
    for n in range(num_assets):
        avg_cov_mat[i,n] = np.average(cov_mat[:,i,n])

avg_corr_mat = cov_to_corr_matrix(avg_cov_mat)


#plot of covariance
fig, ax = plt.subplots(figsize=(8, 8))
correlation_plot(avg_corr_mat, names = names, include_values=True, ax=ax)
plt.show()

#----------------------------------------------------------------------------------- Covariance matrix in bad state

weights = np.array([1/4, 1/4, 1/4,1/4])

eqw_returns = (returns_filtered * weights).sum(axis = 1)
#print(eqw_returns)
rolling_eqw_returns = ((eqw_returns.rolling(30).mean() + 1) ** 365 - 1)


plt.plot(dates,rolling_eqw_returns)
plt.show()


bad_times_returns = rolling_eqw_returns.loc[rolling_eqw_returns <= - 0.3]
bad_time_dates = bad_times_returns.index

returns_bad_daily = returns_filtered.loc[returns_filtered.index.isin(bad_time_dates)]


num_periods = len(returns_bad_daily)
window_size = 365
effective_num_periods = num_periods - window_size
half_life = 365 / 2
time_points = np.arange(1, window_size + 1)


exp_probs_bad = calculate_exponential_decay_probabilities(window_size, time_points, half_life)

cov_mat_bad = np.zeros((effective_num_periods,num_assets,num_assets))
avg_cov_mat_bad = np.zeros((num_assets,num_assets))

for t in range(effective_num_periods):

    cov_mat_bad[t,:,:] = calculate_cov_mat(returns_bad_daily.iloc[t: window_size + t, :].values, probs = exp_probs_bad)

for i in range(num_assets):
    for n in range(num_assets):
        avg_cov_mat_bad[i,n] = np.average(cov_mat_bad[:,i,n])

avg_corr_mat_bad = cov_to_corr_matrix(avg_cov_mat_bad)



#plot of covariance
fig, ax = plt.subplots(figsize=(8, 8))
correlation_plot(avg_corr_mat_bad, names = names, include_values=True, ax=ax)
plt.show()


#----------------------------------------------------------------------------------- Estimating gaussian mixture model

Pi_good = 0.8
eqw_returns_bad = (returns_bad_daily * weights).sum(axis = 1)
eqw_returns = (returns_filtered * weights).sum(axis = 1)

mu_good = np.mean(eqw_returns)
mu_bad = np.mean(eqw_returns_bad)

sigma_good = np.std(eqw_returns)
sigma_bad = np.std(eqw_returns_bad)

x_values = np.arange(-0.0125, 0.0125, 0.00001)
density_g = stats.norm.pdf(x_values, loc=mu_good, scale=sigma_good)
density_b = stats.norm.pdf(x_values, loc=mu_bad, scale=sigma_bad)

density_comb = Pi_good * density_g + (1 - Pi_good) * density_b


plt.plot(x_values, density_g, x_values, density_b, x_values, density_comb)
plt.legend(("Good","Bad",'Mixture'))
plt.show()

plt.plot(x_values, density_comb)
plt.legend(('Mixture'))
plt.show()

