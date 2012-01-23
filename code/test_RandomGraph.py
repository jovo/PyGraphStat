#!/usr/bin/env python
# encoding: utf-8
"""
test.py

Created by Joshua Vogelstein on 2012-01-12.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

import unittest
import RandomGraph as rg
import numpy as np
from nose.tools import *
from scipy import misc

class test(unittest.TestCase):
	def setUp(self):
		pass

		
def test_ErdosRenyi():
	n=10;
	G=rg.ER(n,0.5)
	assert_equal(G.number_of_nodes(),n)
	assert_true(G.number_of_edges() <=misc.comb(n,2,exact=1))
	
def test_SBM():
	k=3 								# number of blocks
	n_min = 2 							# minimum number of vertices per block
	n_max = 10 							# maximum number of vertices per block
	nvec=np.random.random_integers(n_min,n_max,k) # number of vertices per block
	B=np.random.uniform(0,1,(k,k)) 		# probabilities of connections between all blocks
	G=rg.SBM(nvec,B)

	n=nvec.sum()
	assert_equal(G.number_of_nodes(),n)
	assert_true(G.number_of_edges() <=n**2)


if __name__ == '__main__':
	unittest.main()