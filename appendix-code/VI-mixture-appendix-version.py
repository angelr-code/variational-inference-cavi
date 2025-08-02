import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns

def initialize_parameters(n_samples,n_centers):
  """
  Initializes the model parameters given the sample size and the number of clusters.

  Parameters
  ----------
  n_samples: int
    Sample size.

  n_centers: int
    Number of clusters.

  Returns
  --------
  m: numpy.array
    Matrix formed by the variational mean vectors.

  s2: numpy.array
    Vector of variational variances for each cluster.

  phi: numpy.array
    Variational phi vector of probabilities.
  """
  n = n_samples
  k = n_centers

  m = np.random.randn(k,2)*2
  s2 = np.ones(k)*10 # these are variances
  phi = np.ones((n,k))*1/k

  return m, s2, phi


def compute_elbo(X, m, s2, phi, sigma2):
  """
  Computes the ELBO of the model.

  Parameters
  ----------
  X: numpy.array
    Input data matrix (n_samples, n_features).

  m: numpy.array
    Matrix of variational means.

  s2: numpy.array
    Vector of variational variances.

  phi: numpy.array
    Variational phi matrix with responsibilities.

  sigma2: float
    Variance of the generative model.

  Returns
  -------
  elbo: float
    The value of the ELBO bound.
  """
  n,k = phi.shape

  term1 = - k * np.log(2*np.pi*sigma2) - 1/(2*sigma2)*(np.sum(m**2) + 2*np.sum(s2))
  term2 = -n*np.log(k)


  term3 = 0
  for i in range(n):
      for j in range(k):
          diff = X[i] - m[j]
          term3 += phi[i, j] * (-0.5 * np.log(2 * np.pi) - 0.5 * np.sum(diff**2) - s2[j])

  term4 = -np.sum(phi*np.log(phi))
  term5 = -np.sum(np.log(2*np.pi*np.e*s2))

  elbo = term1 + term2 + term3 + term4 + term5

  return elbo

def update_phi(X,m,s2,phi):
    """
    Updates the phi matrix of variational responsibilities.

    Parameters
    ----------
    X: numpy.array
      Input data (n_samples x n_features).

    m: numpy.array
      Current variational means.

    s2: numpy.array
      Current variational variances.

    phi: numpy.array
      Current responsibility matrix.

    Returns
    -------
    phi: numpy.array
      Updated responsibility matrix.
    """

    n, _ = X.shape
    k = m.shape[0]
    phi = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            diff = X[i,:] - m[j,:]                      # (d,)
            sq_norm = np.dot(diff, diff)           # ||x_i - m_k||^2
            var_term = s2[j]               # s_{k}^2
            phi[i, j] = np.exp(-0.5 * sq_norm - 0.5 * var_term)

        # Normalize
        phi[i, :] /= np.sum(phi[i, :])

    return phi


def update_centers(X,phi,sigma2,m,s2):
  """
  Updates the variational means m and variances s2 of the model.

  Parameters
  ----------
  X: numpy.array
    Input data (n_samples x n_features).

  phi: numpy.array
    Matrix of variational responsibilities.

  sigma2: float
    Variance of the generative model.

  m: numpy.array
    Current means.

  s2: numpy.array
    Current variances.

  Returns
  -------
  m: numpy.array
    Updated means.

  s2: numpy.array
    Updated variances.
  """
  n = X.shape[1]
  k = m.shape[0]

  for j in range(k):
    m[j,:] = np.dot(phi[:, j], X)/(1/sigma2 + np.sum(phi[:,j]))
    s2[j] = 1/(1/sigma2 + np.sum(phi[:,j]))

  return m,s2


def cavi(X,n_centers,sigma2, max_iter = 50, plot_start = True):
  """
  Runs the CAVI (Coordinate Ascent Variational Inference) algorithm.

  Parameters
  ----------
  X: numpy.array
    Input data (n_samples x n_features).

  n_centers: int
    Number of clusters.

  sigma2: float
    Variance of the generative model.

  max_iter: int
    Maximum number of iterations.

  plot_start: bool
    If True, it shows a plot with the initial centers. Default: True.

  Returns
  -------
  phi: numpy.array
    Matrix of variational responsibilities.

  m: numpy.array
    Learned variational means.

  s2: numpy.array
    Learned variational variances.

  elbo_values: list
    ELBO values during the iterations.
  """

  n = X.shape[0]
  k = n_centers

  m, s2 , phi = initialize_parameters(n,k)

  if plot_start:
    fig, ax = plt.subplots(figsize = (8,6))
    scatter = ax.scatter(X[:,0], X[:, 1], c = y, cmap='tab10', edgecolor = 'k', s = 60, alpha = 0.8)
    scatter = ax.scatter(m[:,0], m[:,1], c = 'yellow', edgecolor = 'k', s = 100, alpha = 1)
    plt.savefig("inicio.pdf", format="pdf", dpi=300)
    plt.show()
    print("ELBO MONITORING")

  elbo_values = []
  elbo_values.append(compute_elbo(X, m, s2, phi, sigma2))

  for i in range(max_iter):
    phi = update_phi(X,m,s2,phi)
    m, s2 = update_centers(X,phi,sigma2,m,s2)
    elbo_values.append(compute_elbo(X , m , s2, phi, sigma2))

    if i % 5 == 0 and plot_start:
      print(f"ELBO value at iteration {i}: {elbo_values[-1]}")  # ELBO monitoring

  return phi, m, s2, elbo_values


"""Application"""
# Initial data definition
np.random.seed(123)
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std= 1, shuffle= True)


sns.set(style = "darkgrid", context = "notebook")
fig, ax = plt.subplots(figsize = (8,6))
scatter = ax.scatter(X[:,0], X[:, 1], c = y, cmap='tab10', edgecolor = 'k', s = 60, alpha = 0.8)
ax.set_title("Four Normally Distributed Clusters")

plt.show()

#CAVI
phi, m, s2, elbo_values = cavi(X, n_centers=4, sigma2=1, max_iter=50)


# ELBO monitoring
plt.figure(figsize=(8, 5))
plt.xlabel("Iteration", fontsize = 19)
plt.ylabel("ELBO")
plt.grid(True)
plt.legend()
plt.tight_layout()

elbos = np.zeros((10, 51))


for i in range(elbos.shape[0]):
  _,_,_, elbo_values = cavi(X, n_centers=4, sigma2=1, max_iter=50, plot_start=False)
  plt.plot(elbo_values, label="ELBO", color='royalblue')

plt.savefig("elbo.pdf", format = 'pdf', dpi = 300)
plt.show()

# Final result
fig, ax = plt.subplots(figsize = (8,6))
scatter = ax.scatter(X[:,0], X[:, 1], c = y, cmap='tab10', edgecolor = 'k', s = 60, alpha = 0.8)
scatter = ax.scatter(m[:,0], m[:,1], c = 'yellow', edgecolor = 'k', s = 100, alpha = 1)
#plt.savefig("final.pdf", format = 'pdf', dpi = 300)

plt.show()