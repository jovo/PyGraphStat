#!/usr/bin/env python
# encoding: utf-8
"""
test_DPE.py

Created by Joshua Vogelstein on 2012-01-22.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

import unittest
import RandomGraph as rg

class test_DPE(unittest.TestCase):
	def setUp(self):
		pass

	def test_DPE(self):
		G=rg.ER(5,0.1)    
		return G

    
if __name__ == '__main__':
	unittest.main()