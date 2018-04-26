import numpy as np


def naive(series):
	return series.iloc[-1]

def average(series):
	return series.sum()/len(series)

def moving_avg(series, k = None):
	if (k == None) or (len(series) <= k):
		return average(series)
	else:
		return series.iloc[-k:].sum()/k

def expo_smooth(series,alpha):

	if len(series) == 1:
		return alpha*series.iloc[0]
	else:
		return alpha*series.iloc[-1] + (1-alpha)*expo_smooth(series[:-1],alpha)

def double_expo_smooth(series, alpha, beta):

    l = series[:2].values
    b = [l[1]-l[0]]
    y_hat = series[:2].values

    for i in range(2,len(series)+1):
        y = series.iloc[i-1].values
        l = np.append(l, alpha*y + (1-alpha)*y_hat[-1])
        b = np.append(b, beta*(l[-1]-l[-2]) + (1-beta)*b[-1])
        y_hat = np.append(y_hat, l[-1]+b[-1])
    
    return y_hat
