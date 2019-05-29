import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

def generateCauchyData(N, mean):

    return (np.random.standard_cauchy(N) * 100. + mean)

data = generateCauchyData(10000, 42.)
plt.plot(data)
plt.show()

def frequentistCenter(data):

    return np.mean(data)

X = frequentistCenter(data)

print("Sample mean: ", X)

def bayesianCenter(data):

    with pm.Model():
        loc = pm.Uniform('location', lower=-1000., upper=1000.)
        scale = pm.Uniform('scale', lower=0.01, upper=1000.)

        pm.Cauchy('y', alpha=loc, beta=scale, observed=data)

        trace = pm.sample(3000, tune=3000, target_accept=0.92)
        pm.traceplot(trace)
        plt.show()

    return np.mean(trace['location'])

X2 = bayesianCenter(data)

print("Bayesian mode (median, location): ", X2)
