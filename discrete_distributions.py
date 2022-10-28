import numpy as np
from scipy.special import digamma, polygamma, gammaln
import collections
from scipy.integrate import quad


class DiscreteDistribution(object):
    def __init__(self, data, alphabeth):
        self.alphabeth = alphabeth
        self._n_classes = len(alphabeth)

        counter = collections.Counter(data)
        total = sum(counter.values())
        self._probs = {}
        self._counts = {}
        for k in alphabeth:
            self._counts[k] = counter.get(k, 0)
            self._probs[k] = self._counts[k] / total

        assert np.isclose(sum(self._probs.values()), 1)

    def compute_entropy(self, bayes=False, normalize=False):
        norm = np.log(self._n_classes) if normalize else 1
        if bayes:
            return self._compute_bayesian_entropy() / norm
        else:
            return -sum(p * np.log(p) for p in self._probs.values() if p) / norm

    def _compute_bayesian_entropy(self):
        return BayesianEntropyCalculator(self._n_classes, self._counts).entropy

    def _compute_bayesian_entropy_var(self):
        return BayesianEntropyCalculator(self._n_classes, self._counts).entropy_var

    def _compute_probability_uncertainty(self):
        "according to Samengo 2002"
        counter = self._counts
        total = sum(counter.values())
        moment = 2  # second moment
        beta = 1  # uniform prior
        s = self._n_classes  # n classes
        uncertainty = {}
        for k, v in counter.items():
            uncertainty[k] = np.exp(
                gammaln(v + moment + beta)
                + gammaln(total + s * beta)
                - gammaln(v + beta)
                - gammaln(total + s * beta + moment)
            )
        return uncertainty

    def compute_kl_div(self):
        e = 1 / self._n_classes
        return sum(p * (np.log(p) - np.log(e)) for p in self._probs.values() if p)

    def compute_js_div(self):
        e = 1 / self._n_classes
        m = {k: (v + e) / 2 for k, v in self._probs.items()}
        kl_pm = sum(p * (np.log(p) - np.log(m[k])) for k, p in self._probs.items() if p)
        kl_me = sum(p * (np.log(p) - np.log(e)) for p in m.values())
        return (kl_pm + kl_me) / 2


class BayesianEntropyCalculator(object):
    """From Nemenman et al.2002"""

    def __init__(self, n_classes, counts):
        assert n_classes == len(counts)
        self._n_classes = n_classes
        self._log_n_classes = np.log(self._n_classes)
        self._counts = counts
        self._counts_values = np.array([v for v in self._counts.values()])
        self._total = sum(counts.values())
        self._entropy = None
        self._entropy_var = None

    def _exp_entropy(self, beta):
        pseudototal = self._total + beta * self._n_classes
        pseudocounts = self._counts_values + beta
        return (
            digamma(pseudototal + 1)
            - (pseudocounts * digamma(pseudocounts + 1)).sum() / pseudototal
        )

    def _auxiliary_ii(self, nk, ni, pseudototal):
        return (digamma(nk + 1) - digamma(pseudototal + 2)) * (
            digamma(ni + 1) - digamma(pseudototal + 2)
        ) - polygamma(1, pseudototal + 2)

    def _auxiliary_j(self, ni, pseudototal):
        return (
            (digamma(ni + 2) - digamma(pseudototal + 2)) ** 2
            + polygamma(1, ni + 2)
            - polygamma(1, pseudototal + 2)
        )

    def _exp_sq_entropy(self, beta):
        pseudototal = self._total + beta * self._n_classes
        pseudocounts = {k: v + beta for k, v in self._counts.items()}
        res = 0
        for i, ni in pseudocounts.items():
            for k, nk in pseudocounts.items():
                if i == k:
                    res += (ni + 1) * ni * self._auxiliary_j(ni, pseudototal)
                else:
                    res += ni * nk * self._auxiliary_ii(ni, nk, pseudototal)
        return res / (pseudototal * (pseudototal + 1))

    def _hyperprior(self, beta):
        return (
            self._n_classes * polygamma(1, self._n_classes * beta + 1)
            - polygamma(1, beta + 1)
        ) / self._log_n_classes

    def _kernel_entropy(self, beta):
        return self._hyperprior(beta) * self._exp_entropy(beta)

    def _kernel_sq_entropy(self, beta):
        return self._hyperprior(beta) * self._exp_sq_entropy(beta)

    @property
    def entropy(self):
        if self._entropy is None:
            self._entropy = quad(self._kernel_entropy, 0, np.inf)[0]
        return self._entropy

    @property
    def entropy_var(self):
        if self._entropy_var is None:
            sq_entropy = quad(self._kernel_sq_entropy, 0, np.inf)[0]
            self._entropy_var = sq_entropy - self.entropy ** 2
        return self._entropy_var
