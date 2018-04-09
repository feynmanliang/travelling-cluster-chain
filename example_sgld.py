import scipy.stats
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 100
X = np.random.randn(N, 1)*np.sqrt(2)
X[np.where(np.random.randint(2, N, 1) == 1), 0] += 1

def logp(theta, X):
  n1s = np.exp(-0.5*np.power((X - theta[0]), 2)/2)
  n1 = 0
  for i in n1s:
    n1 += np.log(i)
  n1 = np.exp(n1)
  n2s =  np.exp(-0.5*np.power((X - theta[0] - theta[1]), 2)/2)
  n2 = 0
  for i in n2s:
    n2 += np.log(i)
  n2 = np.exp(n2)
  return np.log(0.5*n1 + 0.5*n2)

def dlogp(theta, x):
  return grad(lambda theta: logp(theta, x))(theta)

def logprior(theta):
  return -0.5*np.power(theta[0], 2) / 10 + -0.5*np.power(theta[1], 2)

dlogprior = grad(logprior)

num_samples = 100000
samples = np.zeros(shape=(num_samples, 2))
theta = np.array([1.0, -1.0])
for t in tqdm(range(num_samples)):
  samples[t,:] = theta
  eps = 0.04*np.power((10 + t), -0.55)
  nu = np.random.randn(2) * np.sqrt(eps)

  theta += (eps/2) * (dlogprior(theta) + N*dlogp(theta, X[np.random.randint(N), :])) + nu

plt.subplot(211)
plt.plot(samples)
plt.subplot(212)
plt.scatter(samples[:,0], samples[:,1], alpha=0.1)
plt.savefig('fig.png')
