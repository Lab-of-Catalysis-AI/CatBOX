# -*- coding: utf-8 -*-
#==========================================
# Title:  BaseBO.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
#==========================================

import os
import pickle
import random

import numpy as np


class BaseBO():
    """
    Base class with common operations for BO with continuous and categorical
    inputs
    """

    def __init__(self, objfn, initN, bounds, C, rand_seed=108, debug=False,
                 batch_size=1, init_data=None, **kwargs):
        self.f = objfn  # function to optimise
        self.bounds = bounds  # function bounds
        self.batch_size = batch_size
        self.C = C  # no of categories
        self.initN = initN  # no: of initial points
        self.nDim = len(self.bounds)  # dimension
        self.rand_seed = rand_seed
        self.debug = debug
        self.saving_path = None
        self.kwargs = kwargs
        self.x_bounds = np.vstack([d['domain'] for d in self.bounds
                                   if d['type'] == 'continuous'])
        self.init_data = init_data  # External initial dataset provided

    def initialise(self, seed):
        """Get NxN intial points"""
        data = []
        result = []

        # If external initial dataset is provided, use it directly
        if self.init_data is not None:
            print(f"Using external init data (from CSV file)")
            Zinit = self.init_data.copy()
            yinit = self.f.compute(Zinit, minimize=False)
            data.append(Zinit)
            result.append(yinit)
            return data, result

        np.random.seed(seed)
        random.seed(seed)


        print(f"Creating init data for seed {seed}")
        Xinit = self.generateInitialPoints(self.initN,
                                           self.bounds[len(self.C):])
        hinit = np.hstack(
            [np.random.randint(0, C, self.initN)[:, None] for C in self.C])
        Zinit = np.hstack((hinit, Xinit))
        yinit = self.f.compute(Zinit, minimize=False)

        init_data = {}
        init_data['Z_init'] = Zinit
        init_data['y_init'] = yinit


        data.append(Zinit)
        result.append(yinit)
        return data, result

    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array(
                [np.random.uniform(bounds[b]['domain'][0],
                                   bounds[b]['domain'][1], 1)[0]
                 for b in range(nDim)])
        return Xinit

    def my_func(self, Z):
        Z = np.atleast_2d(Z)
        if len(Z) == 1:
            X = Z[0, len(self.C):]
            ht_list = list(Z[0, :len(self.C)])
            return self.f(ht_list, X)
        else:
            f_vals = np.zeros(len(Z))
            for ii in range(len(Z)):
                X = Z[ii, len(self.C):]
                ht_list = list(Z[ii, :len(self.C)].astype(int))
                f_vals[ii] = self.f(ht_list, X)
            return f_vals

    def save_progress_to_disk(self, *args):
        raise NotImplementedError

    def runTrials(self, budget, n_trail):
        raise NotImplementedError
