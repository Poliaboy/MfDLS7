import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



class GMM:
    def __init__(self, K, n_runs=200):
        self.K = K
        self.n_runs = n_runs
        self.pi = None
        self.mu = None
        self.sigma = None

    def calculate_mean_covariance(self, X, prediction):
        """Calculate means and covariance of different
            clusters from k-means prediction

        Parameters:
        ------------
        prediction: cluster labels from k-means

        X: N*d numpy array data points 

        Returns:
        -------------
        intial_means: for E-step of EM algorithm

        intial_cov: for E-step of EM algorithm

        """
        d = X.shape[1]
        labels = np.unique(prediction)
        self.mu = np.zeros((self.K, d))
        self.sigma = np.zeros((self.K, d, d))
        self.pi = np.zeros(self.K)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)  # returns indices
            self.pi[counter] = len(ids[0]) / X.shape[0]
            self.mu[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.mu[counter, :]
            Nk = X[ids].shape[0]  # number of data points in current gaussian
            self.sigma[counter, :, :] = np.dot(self.pi[counter] * de_meaned.T, de_meaned) / Nk
            counter += 1
        assert np.sum(self.pi) == 1

        return (self.mu, self.sigma, self.pi)

    def _initialise_parameters(self, X):
        """Implement k-means to find startin parameter values.
            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster

        Parameters:
        ------------
        X: numpy array of data points

        Returns:
        ----------
        tuple containing initial means and covariance

        _initial_means: numpy array: (K*d)

        _initial_cov: numpy array: (K,d*d)


        """
        n_clusters = self.K
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, algorithm='auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self.mu, self.sigma, self.pi = self.calculate_mean_covariance(X, prediction)

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def _e_step(self, X):
        N = X.shape[0]
        gamma = np.zeros((N, self.K))
        for k in range(self.K):
            try:
                gamma[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k, :], self.sigma[k])
            except Exception as e:
                print("Error in E-step for component", k, ":", e)
                return None
        gamma_norm = np.sum(gamma, axis=1)[:, np.newaxis]
        gamma /= gamma_norm
        return gamma

    def _m_step(self, X, gamma):
        """Performs M-step of the GMM
        We need to update our priors, our means
        and our covariance matrix.
        Parameters:
        -----------
        X: (N x d), data
        gamma: (N x K), posterior distribution of lower bound
        Returns:
        ---------
        pi: (K)
        mu: (K x d)
        sigma: (K x d x d)
        """
        N = X.shape[0]  # number of objects
        K = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object
        self.mu = np.zeros((self.K, d))
        self.sigma = np.zeros((self.K, d, d))
        self.pi = np.zeros(self.K)

        Nk = np.sum(gamma, axis=0)

        for k in range(self.K):
            self.mu[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nk[k]
            X_minus_mu = X - self.mu[k]
            self.sigma[k] = np.dot(gamma[:, k] * X_minus_mu.T, X_minus_mu) / Nk[k]
            self.pi[k] = Nk[k] / N

        return self.pi, self.mu, self.sigma

    def fit(self, X, epsilon=1e-4):
        N, d = X.shape
        self._initialise_parameters(X)  # Initialize parameters using k-means
        try:
            gamma_old = None
            for run in range(self.n_runs):
                # E-step
                gamma = self._e_step(X)
                # Check for convergence
                if gamma_old is not None and np.linalg.norm(gamma - gamma_old) < epsilon:
                    break
                gamma_old = gamma

                # M-step
                self.pi, self.mu, self.sigma = self._m_step(X, gamma)

        except Exception as e:
            print("An error occurred:", e)
            return None

        print("Fit completed successfully")
        return self

    def predict_func(self, X):
        """
        Returns predicted labels using Bayes Rule to Calculate the posterior distribution.
        """
        proba = self.predict_proba_func(X)
        labels = proba.argmax(axis=1)
        return labels

    def predict_proba_func(self, X):
        """
        Using Bayes Rule to calculate the posterior distribution.
        """
        N, d = X.shape
        post_proba = np.zeros((N, self.K))

        for k in range(self.K):
            post_proba[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k, :], self.sigma[k])

        post_proba /= post_proba.sum(axis=1, keepdims=True)
        return post_proba


def plot_clusters(X, y_true, y_gmm, y_kmeans, title):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, edgecolor='k', cmap='viridis')
    plt.title("True Labels: " + title)

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_gmm, edgecolor='k', cmap='viridis')
    plt.title("GMM Clusters: " + title)

    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, edgecolor='k', cmap='viridis')
    plt.title("K-means Clusters: " + title)

    plt.show()

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10
# Instantiate GMM
gmm = GMM(K=3, n_runs=100)

# For each dataset
for X, y, title in [(X_aniso, y, 'Anisotropic'),
                    (X_varied, y_varied, 'Varied Variance'),
                    (X_filtered, y_filtered, 'Filtered')]:
    # GMM clustering
    gmm = gmm.fit(X)
    y_gmm = gmm.predict_func(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=170)
    y_kmeans = kmeans.fit_predict(X)

    # Plot
    plot_clusters(X, y, y_gmm, y_kmeans, title)
