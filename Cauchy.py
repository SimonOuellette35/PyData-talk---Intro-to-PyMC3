import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

def generateCauchyData(N, mean):

    return (np.random.standard_cauchy(N) * 100. + mean)

data = generateCauchyData(10000, 42.)
plt.plot(data)
plt.show()

def frequentistMean(data):

    return np.mean(data)

X = frequentistMean(data)

print("Sample mean: ", X)

def bayesianMean(data):

    with pm.Model():
        mean = pm.Uniform('mean', lower=-1000., upper=1000.)
        var = pm.Uniform('var', lower=0.01, upper=1000.)

        pm.Cauchy('y', alpha=mean, beta=var, observed=data)

        trace = pm.sample(3000, tune=3000, target_accept=0.92)
        pm.traceplot(trace)
        plt.show()

    return np.mean(trace['mean'])

X2 = bayesianMean(data)

print("Bayesian mean: ", X2)