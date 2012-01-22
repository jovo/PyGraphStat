#!/usr/bin/env python
# encoding: utf-8
"""
test.py

Created by Joshua Vogelstein on 2012-01-12.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""

import unittest
import RandomGraph as rg

class test(unittest.TestCase):
	def setUp(self):
		pass

		
def test_ErdosRenyi():
	G=rg.ER(5,0.1)    
	return G
	

if __name__ == '__main__':
	unittest.main()