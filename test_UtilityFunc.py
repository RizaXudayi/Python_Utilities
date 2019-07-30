# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:33:12 2019

@author: Riza

Test class for PlanarWave class.
The test function members in the relevant classes could be run by the following
command inside Spyder:
    !python -m unittest

"""
import unittest

import numpy as np
pi = np.pi
exp = np.exp
shape = np.shape
reshape = np.reshape
size = np.size
import numpy.linalg as la

from UtilityFunc import UF
uf = UF()

#%% Class definition:

class TestPlanarWave(unittest.TestCase):
    
    def test_nodeNum(self):
        """Test nodeNum function."""
        # Type errors:
        x = np.array([1.0, None, 3.0])[np.newaxis].T
        self.assertRaises(TypeError, uf.nodeNum, x, 2.0)
        
        x = np.array([1.0, 2.0, 3.0])[np.newaxis].T
        self.assertRaises(TypeError, uf.nodeNum, x, None)
        self.assertRaises(TypeError, uf.nodeNum, x, [None])
        
        # Value errors:
        self.assertRaises(ValueError, uf.nodeNum, x, np.array([[1.0, 2.0]]))
        self.assertRaises(ValueError, uf.nodeNum, x, np.array([[1.0, 2.0], [-1.0, 2.0]]))
        
        x = np.array([[1.0, 2.0], [-1.0, -3.0]])
        self.assertRaises(ValueError, uf.nodeNum, x, 2.0)
        self.assertRaises(ValueError, uf.nodeNum, x, [2.0])
        self.assertRaises(ValueError, uf.nodeNum, x, np.array([2.0]))
        self.assertRaises(ValueError, uf.nodeNum, x, np.array([[2.0]]))
        self.assertRaises(ValueError, uf.nodeNum, x, np.array([[2.0, 1.0, 0.0]]))
        
        # Check functionality:
        x = np.array([1.0, 2.0, 3.0])[np.newaxis].T
        self.assertEqual(uf.nodeNum(x, 2.0), [1])
        self.assertEqual(uf.nodeNum(x, [1.0, 2.0]), [0,1])
        self.assertEqual(uf.nodeNum(x, np.array([1.0, 2.0])), [0,1])
        self.assertEqual(uf.nodeNum(x, np.array([[1.0], [2.0]])), [0,1])
        
        x = np.array([[1.0, 2.0], [-1.0, -3.0], [-1.0, 3.0]])
        self.assertEqual(uf.nodeNum(x, np.array([[1.0, 2.0]])), [0])


    def test_Type(self):
        """Test Type() function."""
        
        a = np.array([2])
        self.assertEqual(uf.Type(a[0]), int)
        self.assertEqual(uf.Type(2), int)
        
        a = np.array([2.0])
        self.assertEqual(uf.Type(a[0]), float)
        self.assertEqual(uf.Type(2.0), float)

        a = np.array([2.0j])
        self.assertEqual(uf.Type(a[0]), complex)
        self.assertEqual(uf.Type(2.0j), complex)
        
        self.assertEqual(uf.Type([]), list)


    def test_addNoise(self):
        """Test addNoise() function."""
        
        # Type errors:
        self.assertRaises(TypeError, uf.addNoise, sig=None, delta=0.1)
        self.assertRaises(TypeError, uf.addNoise, [1], 0.1)
        self.assertRaises(TypeError, uf.addNoise, [1.0], None)
        self.assertRaises(TypeError, uf.addNoise, [1.0], [0.1])
        
        # Value errors:
        self.assertRaises(ValueError, uf.addNoise, [1.0], 0.1, 'gamma')
        self.assertRaises(ValueError, uf.addNoise, [1.0], 0.1, 'gaussian', 'distributive')
        
        # Ensure that noise is float for float signal:
        sig = np.random.uniform(size=[10,1])
        _, _, noise = uf.addNoise(sig, 0.1)
        self.assertEqual(uf.Type(noise[0,0]), float)
        _, _, noise = uf.addNoise(sig, 0.1, 'uniform')
        self.assertEqual(uf.Type(noise[0,0]), float)
        _, _, noise = uf.addNoise(sig, 0.1, method='multiplicative')
        self.assertEqual(uf.Type(noise[0,0]), float)
        
        # Ensure that noise is complex for complex signal:
        sig = np.random.uniform(size=[10,1]) + 1.0j*np.random.uniform(size=[10,1])
        _, _, noise = uf.addNoise(sig, 0.1)
        self.assertEqual(uf.Type(noise[0,0]), complex)
        _, _, noise = uf.addNoise(sig, 0.1, 'uniform')
        self.assertEqual(uf.Type(noise[0,0]), complex)
        _, _, noise = uf.addNoise(sig, 0.1, method='multiplicative')
        self.assertEqual(uf.Type(noise[0,0]), complex)




