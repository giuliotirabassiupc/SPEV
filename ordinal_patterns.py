import numpy as np
from scipy.stats import rankdata
from collections import defaultdict
from discrete_distributions import DiscreteDistribution
from itertools import permutations


class OrdinalPattern(object):
    def __init__(self, data, order=3, complexity="KL"):
        self._order = order
        self._complexity_disequilibrium = complexity
        self._alphabeth = self._compute_alphabeth()
        self._repr = self._compute_representation(data)
        self._probs = self._compute_probs()
        self._entropy = self._compute_entropy()
        self._complexity = self._compute_complexity()

    def _compute_representation(self, data):
        raise NotImplementedError("This method is not implemented")

    def _compute_alphabeth(self):
        symbols = np.arange(1, self._order + 1)
        alphabeth = []
        for perm in permutations(symbols):
            alphabeth.append("".join([str(x) for x in perm]))
        alphabeth = sorted(alphabeth)
        return tuple(alphabeth)

    def _compute_probs(self):
        flatten_list = []
        for ll in self._repr:
            flatten_list.extend(ll)
        return DiscreteDistribution(flatten_list, self._alphabeth)

    def _compute_entropy(self):
        return self._probs.compute_entropy()

    def _compute_complexity(self):
        if self._complexity_disequilibrium == "KL":
            return self._entropy * self._probs.compute_kl_div()
        elif self._complexity_disequilibrium == "JS":
            return self._entropy * self._probs.compute_js_div()

    @property
    def entropy(self):
        return self._entropy

    @property
    def complexity(self):
        return self._complexity

    @property
    def probabilities(self):
        return self._probs


def _determine_ordinal_pattern(chunk):
    return "".join(rankdata(chunk, method="ordinal").astype(int).astype(str))


class TemporalOrdinalPattern(OrdinalPattern):
    def __init__(self, timeseries, order=3) -> None:
        if len(timeseries.shape) > 1:
            raise ValueError("The input timeseries must be a vector")
        super().__init__(timeseries, order=order)
        self._transition = self._compute_transition()

    def _compute_transition(self):
        probs = defaultdict(int)
        for ll in self._repr:
            for i, symb in enumerate(ll[:-1]):
                probs[(symb, ll[i + 1])] += 1
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _compute_representation(self, timeseries):
        res = []
        for offset in range(self.order):
            res.append(
                [
                    _determine_ordinal_pattern(timeseries[i : i + self._order])
                    for i in range(offset, timeseries.size, self._order)
                    if i + self._order <= timeseries.size
                ]
            )
        return tuple([tuple(ll) for ll in res])


class SpatialOrdinalPattern(OrdinalPattern):
    def __init__(self, data, order=3, step=1, complexity="KL"):
        if isinstance(order, int):
            self._h_order = order
            self._v_order = order
            self._pattern_order = order * order
        else:
            self._h_order, self._v_order = order
            self._pattern_order = self._h_order * self._v_order
        self._step = step
        super().__init__(data, self._pattern_order, complexity=complexity)

    def _compute_representation(self, spatial_field):
        if len(spatial_field.shape) != 2:
            raise ValueError("Input spatial field must be 2D")
        max_i_diff = self._v_order * self._step
        max_j_diff = self._h_order * self._step
        res = []
        for i in range(spatial_field.shape[0]):
            max_i = i + max_i_diff
            if max_i > spatial_field.shape[0]:
                continue
            parent_chunk = spatial_field[i : max_i : self._step, :]
            for j in range(spatial_field.shape[1]):
                max_j = j + max_j_diff
                if max_j > spatial_field.shape[1]:
                    continue
                chunk = parent_chunk[:, j : max_j : self._step]
                assert chunk.shape == (self._v_order, self._h_order)
                if np.isnan(chunk).any():
                    continue
                res.append(_determine_ordinal_pattern(chunk.ravel()))
        return (tuple(res),)
