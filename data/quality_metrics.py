from scipy import stats

def statistical_metrics(x):
    x = x[:]
    nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(x, nan_policy='omit')
    differential_entropy = stats.differential_entropy(x, method='vasicek', nan_policy='omit')
    return *minmax, mean, variance, skewness, kurtosis, differential_entropy