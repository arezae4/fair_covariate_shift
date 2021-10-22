import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from KDEpy import NaiveKDE


def create_shift(
    data,
    src_split=0.4,
    alpha=1,
    beta=2,
    kdebw=0.3,
    eps=0.001,
):
    """
    Creates covariate shift sampling of data into disjoint source and target set.

    Let \mu and \sigma be the mean and the standard deviation of the first principal component retrieved by PCA on the whole data.
    The target is randomly sampled based on a Gaussian with mean = \mu and standard deviation = \sigma.
    The source is randomly sampled based on a Gaussian with mean = \mu + alpha and standard devaition = \sigma / beta

    data: [m, n]
    alpha, beta: the parameter that distorts the gaussian used in sampling
                   according to the first principle component
    output: source indices, target indices, ratios based on kernel density estimation with bandwidth = kdebw and smoothed by eps
    """
    m = np.shape(data)[0]
    source_size = int(m * src_split)
    target_size = source_size

    pca = PCA(n_components=2)
    pc2 = pca.fit_transform(data)
    pc = pc2[:, 0]
    pc = pc.reshape(-1, 1)

    pc_mean = np.mean(pc)
    pc_std = np.std(pc)

    sample_mean = pc_mean + alpha
    sample_std = pc_std / beta

    # sample according to the probs
    prob_s = norm.pdf(pc, loc=sample_mean, scale=sample_std)
    sum_s = np.sum(prob_s)
    prob_s = prob_s / sum_s
    prob_t = norm.pdf(pc, loc=pc_mean, scale=pc_std)
    sum_t = np.sum(prob_t)
    prob_t = prob_t / sum_t

    source_ind = np.random.choice(
        range(m), size=source_size, replace=False, p=np.reshape(prob_s, (m))
    )

    pt_proxy = np.copy(prob_t)
    pt_proxy[source_ind] = 0
    pt_proxy = pt_proxy / np.sum(pt_proxy)
    target_ind = np.random.choice(
        range(m), size=target_size, replace=False, p=np.reshape(pt_proxy, (m))
    )

    assert np.all(np.sort(source_ind) != np.sort(target_ind))
    src_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[source_ind, :]
    )
    trg_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[target_ind, :]
    )

    ratios = src_kde.p(pc2, eps) / trg_kde.p(pc2, eps)
    print("min ratio= {:.5f}, max ratio= {:.5f}".format(np.min(ratios), np.max(ratios)))

    return source_ind, target_ind, ratios


class KDEAdapter:
    def __init__(self, kde=NaiveKDE(kernel="gaussian", bw=0.3)):
        self._kde = kde

    def fit(self, sample):
        self._kde.fit(sample)
        return self

    def pdf(self, sample):
        density = self._kde.evaluate(sample)
        return density

    def p(self, sample, eps=0):
        density = self._kde.evaluate(sample)
        return (density + eps) / np.sum(density + eps)
