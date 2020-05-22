#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:26:23 2018

@author: sudhir
"""
# =============================================================================
# Importing the dataset
# =============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# =============================================================================
# Importing the dataset
# =============================================================================
movies = pd.read_csv('ml-1m/movies.dat',sep='::', header = None, 
                     engine = 'python', encoding= 'latin-1')
users = pd.read_csv('ml-1m/users.dat',sep='::', header = None, 
                     engine = 'python', encoding= 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::', header = None, 
                     engine = 'python', encoding= 'latin-1')

#Preapare the traing set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter= '\t')
test_set = np.array(test_set, dtype='int')

