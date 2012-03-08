'''
Tests for adjacency module.

I hope this is the right place to put this.
Contains the class TestRandomGraph which tests each class in RandomGraph module.

Created on Sep 21, 2011

@author: dsussman
'''
import unittest
import adjacency as rg
import numpy as np


class TestRandomGraph(unittest.TestCase):
    """Class to test the RandomGraph module"""
    
    
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testERGraph(self):
        """Run the tests associated with ERGraph"""
        print 'Run through various sizes and graph types and check them'
        p = .01
        for n in xrange(1,10):
            G = rg.ERGraph(n*10,p,directed=True,loopy=False)
            G.check_graph()
            G = rg.ERGraph(n*10,p,directed=False,loopy=False)
            G.check_graph()
            G = rg.ERGraph(n*10,p,directed=True,loopy=True)
            G.check_graph()
            G = rg.ERGraph(n*10,p,directed=False,loopy=True)
            G.check_graph()
        
        n=10000
        G = rg.ERGraph(n,p,directed=True,loopy=True)
            
        print 'Verify RareEventExceptions can be raised'
        with self.assertRaises(rg.RareEventException):
            G.check_rare_event(1)
            
        print 'Checking for rare events to make sure random generation is working'
        try:
            G.check_rare_event(.0001)
        except rg.RareEventException as rare:
            print rare.message
            print 'If this happens again we are in trouble.\nHere goes nothing...'
            G.check_rare_event(.0001)
    
    def testSBMGraph(self):
        """Run the tests associated with SBMGraphs
        
        #TODO: needs to be written"""
        P = .1*np.eye(3)+.1*np.ones((3,3))
        rho = np.array((.2,.3,.5))
        
        for n in xrange(1,10):
            G = rg.SBMGraph(n*10,P,rho,directed=True,loopy=False)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()