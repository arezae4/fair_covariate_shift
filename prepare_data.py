import numpy as np
import pandas as pd


def prepare_compas(normalized=True):

    filePath = "datasets/IBM_compas/"
    dataA = pd.read_csv(
        filePath + "IBM_compas_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_compas_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_compas_X.csv", sep="\t", index_col=0)
    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_german(normalized=True):
    filePath = "datasets/A,Y,X/IBM_german/"

    dataA = pd.read_csv(
        filePath + "IBM_german_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_german_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_german_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_drug(normalized=True):
    filePath = "datasets/drug/"

    dataA = pd.read_csv(filePath + "drug_A.csv", sep="\t", index_col=0, header=None)
    dataY = pd.read_csv(filePath + "drug_Y.csv", sep="\t", index_col=0, header=None)
    dataX = pd.read_csv(filePath + "drug_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_arrhythmia(normalized=True):
    filePath = "datasets/arrhythmia/"

    dataA = pd.read_csv(
        filePath + "arrhythmia_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "arrhythmia_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "arrhythmia_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def normalize(X):
    for c in list(X.columns):
        if X[c].min() < 0 or X[c].max() > 1:
            mu = X[c].mean()
            s = X[c].std(ddof=0)
            X.loc[:, c] = (X[c] - mu) / s
    return X
