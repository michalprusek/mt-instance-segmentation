#!/usr/bin/env python3
"""Fit L-hand (the S1 comparator C2) on synthetic links only and freeze it.   a_fit_hand.py TRAIN.npz OUT.json"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
z = np.load(sys.argv[1], allow_pickle=True); X, y = z["feats"].astype(float), z["y"].astype(int)
mu, sd = X.mean(0), X.std(0) + 1e-6
lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y)
json.dump(dict(features=["mean_logit", "min_logit", "gap", "turn", "pair_cost"], mu=mu.tolist(), sd=sd.tolist(), coef=lr.coef_[0].tolist(),
               intercept=float(lr.intercept_[0]), _fit="synthetic links only: runs/lpc/s1_train384.npz (52 894 links, class_weight balanced), 2026-10-06; write-once"),
          open(sys.argv[2], "w"), indent=1)
print("coef", dict(zip(["mean_logit", "min_logit", "gap", "turn", "pair_cost"], np.round(lr.coef_[0], 3))), "intercept", round(float(lr.intercept_[0]), 3))
