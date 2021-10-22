import numpy as np
import pandas as pd
import argparse


from fair_covariate_shift import eopp_fair_covariate_shift_logloss
from create_shift import create_shift
import prepare_data

data2prepare = {
    "compas": prepare_data.prepare_compas,
    "german": prepare_data.prepare_german,
    "drug": prepare_data.prepare_drug,
    "arrhythmia": prepare_data.prepare_arrhythmia,
}

dataset2reg = {
    "compas": 0.001,
    "german": 0.01,
    "drug": 0.001,
    "arrhythmia": 0.01,
}

dataset2eps = {
    "compas": 0.001,
    "german": 0.001,
    "drug": 0.001,
    "arrhythmia": 0.001,
}

sample_size_ratio = 0.4


def load_dataset(dataset, alpha, beta, kdebw, epsilon):
    dataA, dataY, dataX = data2prepare[dataset]()
    data = pd.concat([dataA, dataX], axis=1).values  # include A in features
    tr_idx, ts_idx, ratios = create_shift(
        data,
        src_split=sample_size_ratio,
        alpha=alpha,
        beta=beta,
        kdebw=kdebw,
        eps=epsilon,
    )
    tr_X, tr_ratio = dataX.iloc[tr_idx, :], ratios[tr_idx]
    ts_X, ts_ratio = dataX.iloc[ts_idx, :], ratios[ts_idx]
    tr_A, tr_Y = dataA.iloc[tr_idx].squeeze(), dataY.iloc[tr_idx].squeeze()
    ts_A, ts_Y = dataA.iloc[ts_idx].squeeze(), dataY.iloc[ts_idx].squeeze()

    dataset = dict(
        X_src=tr_X.values,
        A_src=tr_A.values,
        Y_src=tr_Y.values,
        ratio_src=tr_ratio,
        X_trg=ts_X.values,
        A_trg=ts_A.values,
        Y_trg=ts_Y.values,
        ratio_trg=ts_ratio,
    )

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help='Dataset name : ["compas","german","drug","arrhythmia"].',
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=False,
        default=1,
        help="number of random shuffle runs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=1,
        help="Shift the Gaussian mean -> mean + alpha in sampling of covariates.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=2,
        help="Scale the Gaussian std -> std / beta in sampling of covariates.",
    )
    parser.add_argument(
        "--mu_range",
        type=float,
        required=False,
        nargs="+",
        default=[-1.5, 1.5],
        help="The search range for \mu - the fairness penalty weight.",
    )

    args = parser.parse_args()
    n = args.repeat
    dataset = args.dataset
    alpha = args.alpha
    beta = args.beta
    C = dataset2reg[dataset]
    eps = dataset2eps[dataset]
    kdebw = 0.3  # KDE bandwidth
    mu_range = args.mu_range
    errs, violations = [], []
    for i in range(n):
        print(
            "------------------------------- {} sample {:d} / {:d}, shift parameters: alpha = {}, beta = {}---------------------------------".format(
                dataset, i + 1, n, alpha, beta
            )
        )
        sample = load_dataset(dataset, alpha, beta, kdebw=kdebw, epsilon=eps)
        h = eopp_fair_covariate_shift_logloss(
            verbose=1, tol=1e-7, random_initialization=False
        )
        h.trg_grp_marginal_matching = True
        h.C = C
        h.max_epoch = 3
        h.max_iter = 3000
        h.tol = 1e-7
        h.random_start = True
        h.verbose = 1
        h.fit(
            sample["X_src"],
            sample["Y_src"],
            sample["A_src"],
            sample["ratio_src"],
            sample["X_trg"],
            sample["A_trg"],
            sample["ratio_trg"],
            mu_range=mu_range,
        )
        err = 1 - h.score(
            sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
        )
        violation = abs(
            h.fairness_violation(
                sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
            )
        )
        errs.append(err)
        violations.append(violation)
        print(
            "Test  - prediction_err : {:.3f}\t fairness_violation : {:.3f} ".format(
                err, violation
            )
        )
        print("Mu = {:.4f}".format(h.mu))
        print("")

    print(
        "------------------------------- Summary: {}, {:d} samples, shift parameters: alpha = {}, beta = {}---------------------------------".format(
            dataset, n, alpha, beta
        )
    )
    errs = np.array(errs, dtype=float)
    violations = np.array(violations, dtype=float)
    print(
        "Test  - prediction_err : {:.3f} \u00B1 {:.3f} \t fairness_violation : {:.3f} \u00B1 {:.3f} ".format(
            errs.mean(),
            1.96 / np.sqrt(n) * errs.std(),
            violations.mean(),
            1.96 / np.sqrt(n) * violations.std(),
        )
    )
