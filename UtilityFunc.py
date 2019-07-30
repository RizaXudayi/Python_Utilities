# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:11:51 2018

@author: Riza
"""

#%% Modules:

import os
import shutil
import numbers
import warnings

import numpy as np
import numpy.linalg as la
shape = np.shape
size = np.size
reshape = np.reshape

import scipy.sparse as sparse
from scipy.linalg import block_diag

import csv

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

#%% Utility Function Class:

class UF():
    """Class to define custom functions to facilitate using Python."""

    def isnumber(self, x):
        """Checks if x contains all numbers."""
        if type(x)==list:
            x = self.unpackList(x)
            for x1 in x:
                if type(x1)==np.ndarray:
                    if not self.arrayIsnumber(x1): return False
                elif not isinstance(x1, numbers.Number):
                    return False
            return True
        
        elif type(x)==np.ndarray:
            return self.arrayIsnumber(x)
        
        else:
            return isinstance(x, numbers.Number)
    
    
    def arrayIsnumber(self, x):
        """Checks if an array contains all numbers."""
        if not type(x)==np.ndarray:
            raise ValueError('\'x\' must be a numpy array!')
            
        x = reshape(x, size(x))
        for x1 in x:
            if not isinstance(x1, numbers.Number):
                return False
        return True



    def isempty(self, x):
        """Checks if x is empty."""
        if type(x)==list and len(x)>0:
            return False
        elif type(x)==dict and len(x)>0:
            return False
        elif not (type(x)==list or type(x)==dict) and size(x)>0:
            return False
        else:
            return True
        
        
    def isnone(self, x):
        """Checks if x is none. Returns true even if a single element is None."""
        
        if self.isempty(x):
            return False
        
        elif type(x)==list:
            x = self.unpackList(x)
            for i in range(len(x)):
                if type(x[i]).__module__==np.__name__ and size(x[i])>1 and (x[i]==None).any():
                    return True
                elif type(x[i]).__module__==np.__name__ and size(x[i])==1 and x[i]==None:
                    return True
                elif type(x[i]).__module__==np.__name__:
                    continue
                elif x[i]==None:
                    return True
            return False
        
        elif type(x).__module__==np.__name__ and size(x)>1:
            return (x==None).any()
        
        else:
            return x==None
        
        
    def unpackList(self, x):
        """
        Retruns elements of a list with arbitraty structure, i.e., a list of 
        lists, in a single list.
        """
        # Handle error:
        if type(x)!=list:
            raise ValueError('Input argument must be a list!')
            
        outList = []
        for i in range(len(x)):
            if type(x[i])==list:
                outList.extend(self.unpackList(x[i]))
            elif not self.isempty(x[i]):
                outList.append(x[i])
        
        return outList
    
    
    def vstack(self, tup):
        """
        Function to stack tensors vertically. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
                
        return np.vstack(tup2)
    
    
    def hstack(self, tup):
        """
        Function to stack tensors horizontally. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
        return np.hstack(tup2)


    def nodeNum(self, x, val):
        """
        Function to find closest value in the vector 'x' to value 'val'.
        
        Inputs:
            x [n x dim]: column matrix of values
            val [m x dim]: vector of values to be found in 'x'
        """
        # Handle errors:
        if not type(x) in [list, np.ndarray]:
            raise TypeError('\'x\' must be a column matrix!')
        if type(x) is list: x = np.array(x)
        if not size(x.shape)==2: raise ValueError('\'x\' must be a column matrix!')
        dim = x.shape[1]
        
        if not self.isnumber(val):
            raise TypeError('entries of val must be a numbers!')
        if isinstance(val, numbers.Number): val = np.array([[val]])
        elif type(val) is list: val = np.array(val)
        
        vshape = val.shape
        if dim==1 and size(vshape)==2 and not vshape[1]==1:
            raise ValueError('dimension inconsistant!')
        elif dim==1: val=self.column(val)
        elif not size(vshape)==2 or not vshape[1]==dim: raise ValueError('dimension inconsistant!')
        
        ind = []
        for i in range(val.shape[0]):
            ind.append(np.argmin(la.norm(x-val[i, :], axis=1)))
        
        return ind


    def pairMats(self, mat1, mat2):
        """
        Utility function to pair matrices 'mat1' and 'mat2' by tiling 'mat2' and
        repeating rows of 'mat1' for each tile of 'mat1'.
        
        Inputs:
            mat1 [n1xm1]
            mat2 [n2xm2]
            
        Output:
            MAT [(n1*n2)x(m1+m2)]
        """
        # Error handling:
        if self.isempty(mat1):
            return mat2
        elif self.isempty(mat2):
            return mat1
        
        # Matrix dimensions:
        sh1 = shape(mat1)
        sh2 = shape(mat2)
        
        # Repeat one row of the first matrix per tile of second matrix:
        ind = np.arange(0, sh1[0])[np.newaxis].T
        ind = np.tile(ind, reps=[1, sh2[0]])
        ind = reshape(ind, newshape=sh1[0]*sh2[0])
        MAT1 = mat1[ind]
        
        # Tile second matrix:
        MAT2 = np.tile(mat2, reps=[sh1[0],1])
        
        return np.hstack([MAT1, MAT2])
            

    def rejectionSampling(self, func, smpfun, dof, dofT=None):
        """
        Function to implement the rejection sampling algorithm to select 
        points according to a given loss function. If more than one loss function
        is used in 'func', dofT determines the number of nodes that belong to 
        each one of them.
        
        Inputs:
            func: function handle to determine the loss value at candidate points
            smpfun: function to draw samples from
            dof [mx1]: number of samples to be drawn for each segment
            dofT [mx1]: determines the segment length in the samples and function values
        """
        # Error handling:
        if isinstance(dof, numbers.Number) and not self.isnone(dofT):
            raise ValueError('\'dofT\' must be None for scalar \'dof\'')
        elif isinstance(dof, numbers.Number):
            dof = [dof]
        m = len(dof)
            
        if m>1 and self.isnone(dofT):
            raise ValueError('\'dofT\' must be provided when \'dof\' is a list!')
        
        # Rejection sampling procedure:
        maxfunc = lambda x,i: np.max(x)
        fmax = self.listSegment(func(), dofT, maxfunc)                  # maximum function values over the uniform grid
        def rejecSmp(val, i):
            """Function for rejection sampling to be assigned to listSegment()."""
            nt = len(val)                                               # number of samples
            
            # Uniform drawing for each sample to determine its rejection or acceptance:
            uniformVal = np.random.uniform(size=[nt,1])
            
            # Rejection sampling:
            ind = uniformVal < (val/fmax[i])                            # acceptance criterion
            return reshape(ind, nt)
            
        # Initialization:
        ns = [0 for i in range(m)]                                      # number of samples
        inpuT = [[] for i in range(m)]                                  # keep accepted samples
        flag = True
        while flag:
            # draw new samples:
            samples = smpfun()
            smpList = self.listSegment(samples, dofT)
            
            # Function value at randomly sampled points:
            val = func(samples)
            
            # Rejection sampling for each segment:
            ind = self.listSegment(val, dofT, rejecSmp)                 # accepted indices
            
            flag = False                                                # stopping criterion
            for i in range(m):
                inpuTmp = smpList[i][ind[i]]                            # keep accepted samples
                inpuT[i] = self.vstack([inpuT[i], inpuTmp])             # add to previously accepted samples
                ns[i] += np.sum(ind[i])                                 # update the number of optimal samples
                if not flag and ns[i]<dof[i]: flag=True
            
        for i in range(m):
            inpuT[i] = inpuT[i][:dof[i],:]                              # keep only 'dof' samples
        
        return np.vstack(inpuT)                                         # stack all samples together
        
        
    def listSegment(self, vec, segdof, func=None):
        """
        This function segemnts a vector of values into smaller pieces stored
        in a list and possibly apply 'func' to each segment separately.
        
        Inputs:
            vec [n x dim]: vector to be segmented
            segdof [mx1]: segmentation nodes (each entry specifies the NUMBER 
                   of nodes in one segment)
            func: function to be applied to segments separately - this function
                should accept a list and its index in the original list
        """
        n = len(vec)
        
        # Error handling:
        if self.isnone(segdof) and self.isnone(func):
            return [vec]
        elif self.isnone(segdof):
            return [func(vec,0)]
        elif isinstance(segdof, numbers.Number):
            segdof = [segdof]
            m = 1
        else:
            m = len(segdof)
            
        if segdof[-1]>n:
            raise ValueError('\'segdof\' is out of bound!')
            
        # Segmentation:
        outVec = [[] for i in range(m)]
        ind = 0
        for i in range(m):
            if not self.isnone(func):
                outVec[i] = func(vec[ind:(ind+segdof[i])][:], i)
            else:
                outVec[i] = vec[ind:(ind+segdof[i])][:]
            ind += segdof[i]
            
        # Add the remainder if it exists:
        if ind<n and not self.isnone(func):
            outVec.append( func(vec[ind:], i) )
        elif ind<n:
            outVec.append(vec[ind:])
        
        return outVec
        
        
    def reorderList(self, x, ind):
        """Reorder the entries of the list 'x' according to indices 'ind'."""
        n = len(x)
        
        # Error handling:
        if not len(ind)==n:
            warnings.warn('length of the indices is not equal to the length of the list!')
        if not self.isnumber(ind):
            raise ValueError('\'ind\' must be a list of integers!')
            
        if type(ind)==np.ndarray:
            ind = reshape(ind, n)
            
        return [x[i] for i in ind]
            
        
    def buildDict(self, keys, values):
        """Build a dict with 'keys' and 'values'."""
        n = len(keys)
        
        # Error handling:
        if not len(values)==n:
            raise ValueError('length of the keys and values must match!')
            
        mydict = {}
        for i in range(n):
            mydict[keys[i]] = values[i]
            
        return mydict
        
    
    def l2Err(self, xTrue, xApp):
        """Function to compute the normalized l2 error."""
        
        # Preprocessing:
        if sparse.issparse(xTrue): xTrue = xTrue.todense()
        if sparse.issparse(xApp): xApp = xApp.todense()
        n = size(xTrue)
        if size(shape(xTrue))==1:
            xTrue = reshape(xTrue, [n,1])
        if not size(xApp)==n:
            raise ValueError('\'xTrue\' and \'xApp\' must have the same shape!')
        elif size(shape(xApp))==1:
            xApp = reshape(xApp, [n,1])
        
        return la.norm(xTrue-xApp)/la.norm(xTrue)
        
        
    def clearFolder(self, folderpath):
        """Function to remove the content of the folder specified by 'folderpath'."""
        
        if self.isempty(os.listdir(folderpath)): return
        
        # Make sure that the call to this function was intended:
        while True:
            answer = input('clear the content of the folder? (y/n)\n')
            if answer.lower()=='y' or answer.lower()=='yes':
                break
            elif answer.lower()=='n' or answer.lower()=='no':
                return
        
        for file in os.listdir(folderpath):
            path = os.path.join(folderpath, file)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(e)
        
        
    def copyFile(self, filename, folderpath):
        """
        Function to backup the operator settings for later reference.
        Inputs:
            filename: name of the current operator file
            folderpath: the destination folder path
        """
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The file does not exist!')
        
        shutil.copy2(filename, folderpath)      # copy the file


    def polyArea(self, x, y=None):
        """
        Function to compute the area of a polygon using Shoelace formula.
        
        Inputs:
            x: vector of first coordinates or all coordinates in columns
            y: vector of second coordinates
        """
        if self.isnone(y) and not shape(x)[1]==2:
            raise ValueError('input must be 2d!')
        elif self.isnone(y):
            y = x[:,0]
            x = x[:,1]
        elif len(shape(x))>1 and not shape(x)[1]==1:
            raise ValueError('\'x\' must be a 1d vector of first coordinates!')
        elif not len(x)==len(y):
            raise ValueError('\'x\' and \'y\' must be the same length!')
        else:
            x = reshape(x, len(x))
            y = reshape(y, len(x))
        
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
        
    def mergeDict(self, dictList): 
        """
        Function to merge a list of dictionaries into one dictionary.
        
        Input: list of dictionaries
        """
        # Error handling:
        if not type(dictList)==list:
            raise ValueError('input must be a list of dictionaries!')
            
        if len(dictList)==1:
            return dictList[0]
        elif len(dictList)==2:
            return {**dictList[0], **dictList[1]}
        else:
            return {**dictList[0], **self.mergeDict(dictList[1:])}
            
        
    def csvRead(self, filename):
        """Function to read csv file into a list."""
        
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The csv file does not exist!')
        
        with open(filename, 'r') as file:
          reader = csv.reader(file)
          return list(reader)
        
        
    def column(self, vec):
        """Takes in a vector and returns a column vector."""
        if type(vec) is list: vec = np.array(vec)
        sh1 = vec.shape
        n = vec.size
        if sh1==(n,1): return vec
        elif sh1==(1,n): return vec.T
        elif sh1==(n,): return vec.reshape([n,1])
        else: raise ValueError('\'vec\' must be a vector!')
        
        
    def blkdiag(self, tup, empty=True):
        """
        Constructs the block-diagonal matrix of matrices in tuple 'tup'.
        
        Inputs:
            tup: tuple of matrices to used for block-diagonalization
            empty: if True remove potential empty entries from 'tup'
        """
        if empty: tup = self.rmEmptyList(tup)
        if tup==[]:       return []
        elif len(tup)==1: return np.array(tup[0])
        else:
            return block_diag(tup[0], self.blkdiag(tup[1:], empty=False))
        
        
    def rmEmptyList(self, val):
        """Removes empty entries from given list."""
        if type(val) is not list: raise ValueError('\'val\' must be a list!')
        if val==[]: return []
        
        val1 = []
        for v in val:
            if not self.isempty(v): val1.append(v)
        return val1
        
    
    def Type(self, x):
        """
        Function to return variable types considering different types in Python
        that essentially refer to the same thing from partical point of view.
        """
        # Type: integer
        if type(x) in [int, np.int, np.int0, np.int8, np.int16, np.int32, np.int64]:
            return int
        
        # Type: float
        if type(x) in [float, np.float, np.float16, np.float32, np.float64]:
            return float
        
        # Type: complex
        if type(x) in [complex, np.complex, np.complex64, np.complex128]:
            return complex
        
        return type(x)
        
        
    def addNoise(self, sig, delta, distn='Gaussian', method='additive'):
        """
        Function to generate simulated noisy data given a signal.
        
        Inputs:
            sig [nx1]: column vector of signal values
            delta: noise variation:
                Gaussian: std of the distn
                uniform: range of the distn
            distn: distribution of the noise: Gaussian or uniform
            method:
                additive: add noise to components
                multiplicative: add noise proportional to signal magnitude
                
        Outputs:
            sig: noise-corrputed signal
            SNR: signal-to-noise ratio
            noise: added noise vector
        """
        # Error handling:
        if not self.isnumber(sig): raise TypeError('signal must contain only numbers!')
        sig = self.column(sig)
        if self.Type(sig[0,0]) is float: Float = True
        elif self.Type(sig[0,0]) is complex: Float = False
        else: raise TypeError('entries of \'sig\' must be float or complex!')
        
        if not self.isnumber(delta): raise TypeError('\'delta\' must be a number!')
        if self.Type(delta) is float and delta<0.0:
            raise ValueError('\'delta\' must be a positive number!')
        if Float and self.Type(delta) is list:
            raise TypeError('\'delta\' must be a number!')
        if self.Type(delta) is list and not len(delta)==2:
            raise ValueError('\'delta\' must have exactly two components!')
        if self.Type(delta) is list and (delta[0]<0.0 or delta[1]<0.0):
            raise ValueError('components of \'delta\' must be positive numbers!')
            
        if self.Type(distn) is not str: raise TypeError('\'distn\' must be a string!')
        if not distn.lower() in ['gaussian', 'uniform']: raise ValueError('unknown distribution!')
        
        if self.Type(method) is not str: raise TypeError('\'method\' must be a string!')
        if not method.lower() in ['additive', 'multiplicative']: raise ValueError('unknown method!')
        
        # Pre-processing:
        n = len(sig)
        if not Float and self.Type(delta) is float:
            delta = [delta]*2
            
        # Generate standard random noise:
        if distn.lower()=='gaussian':  noise = np.random.normal(size=[n,2])
        elif distn.lower()=='uniform': noise = np.random.uniform(-1.0, 1.0, size=[n,2])
            
        # Scale appropriately:
        if method.lower()=='additive':
            noise = delta*noise
            if Float: noise = noise[:,0:1]
            else:     noise = noise[:,0:1] + 1j*noise[:,1:2]
            
        elif method.lower()=='multiplicative':
            if Float: noise = sig*delta*noise[:,0:1]
            else:     noise = delta[0]*sig.real*noise[:,0:1] + 1.0j*delta[1]*sig.imag*noise[:,1:2]
        
        SNR = 20*np.log10(la.norm(sig)/la.norm(noise))
        sig2 = sig + noise
        
        return sig2, SNR, noise
        
        
    def stem3(self, x, y):
        """
        Function to stem plot in 3d.
        
        Inputs:
            x [nx2]: coordinates in 2d plane
            y [nx1]: corresponding values
        """
        # Error handling:
        if self.Type(x) is not np.ndarray: raise TypeError('\'x\' must be a column matrix!')
        if not size(x.shape)==2: TypeError('\'x\' must be a column matrix!')
        n, d = x.shape
        if not d==2: raise ValueError('\'x\' must have two columns!')
        
        if self.Type(y) is not np.ndarray: raise TypeError('\'y\' must be a column matrix!')
        y = y.reshape([-1])
        if not len(y)==n: raise ValueError('length of \'y\' does not match \'x\'!')
        
        # Figure:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        x1 = x[:,0]
        x2 = x[:,1]
        for x1i, x2i, yi in zip(x1, x2, y):
            line=art3d.Line3D(*zip((x1i, x2i, 0), (x1i, x2i, yi)), marker='o', markevery=(1, 1))
            ax.add_line(line)
            
        # Set the limits:
        delta = 0.2
        
        x1min = np.min(x1)
        x1max = np.max(x1)
        x1min -= delta*(x1max-x1min)
        x1max += delta*(x1max-x1min)
        
        x2min = np.min(x2)
        x2max = np.max(x2)
        x2min -= delta*(x2max-x2min)
        x2max += delta*(x2max-x2min)
        
        ymin = np.min(y)
        ymax = np.max(y)
        ymin -= delta*(ymax-ymin)
        ymax += delta*(ymax-ymin)
        
        print(x1min)
        print(x1max)
        print(x2min)
        print(x2max)
        print(ymin)
        print(ymax)
        
        ax.set_xlim3d(x1min, x1max)
        ax.set_ylim3d(x2min, x2max)
        ax.set_zlim3d(ymin, ymax)
        ax.view_init(elev=50., azim=35)
        
        
    def stepFun(self, cellVal, cellLims, points):
        """
        Function to select the cell that each point belongs to and assign the value
        of that cell to the point.
        
        Inputs:
            cellVal [nx1]: values at the centers of cells
            cellLims [nx2xd]: list of 2xd matrices containing the lower- and upper-bounds
                of the cells
            points [mxd]: points to be assigned to cells
            
        Output:
            val [mx1]: value of the step function at each point
        """
        
        # Error handling:
        if not self.Type(cellVal) in [list, np.ndarray]:
            raise TypeError('\'cellVal\' must be an array containing step values at the cells!')
        if self.Type(cellVal) is list: cellVal = np.array(cellVal)
        cellVal = cellVal.reshape([-1])
        n = len(cellVal)            # number of cells
        
        if not self.Type(cellLims) in [list, np.ndarray]:
            raise TypeError('\'cellLims\' must be an array containing the lower- and upper-bounds of the cells!')
        if self.Type(cellLims) is list: cellLims = self.vstack(cellLims)
        if not cellLims.shape[:2]==(n,2):
            raise ValueError('\'cellLims\' must be a %ix2xd dimensional matrix!' % n)
        dim = cellLims.shape[2]    # space dimension
        
        if not self.Type(points) in [list, np.ndarray]:
            raise TypeError('\'points\' must be an array containing the inquiry points!')
        if self.Type(points) is list: cellLims = self.vstack(points)
        if not cellLims.shape[1]==dim:
            raise ValueError('\'points\' must exactly have two columns!')
        m = len(points)             # number of inquiry points
        
        val = np.zeros([m,1])
        for i in range(n):          # loop over cells
            lim = cellLims[i]
            ind = np.ones(m)
            for d in range(dim):    # loop over dimensions
                ind *= (lim[0,d]<=points[:,d])*(points[:,d]<=lim[1,d])
            
            # Assign the value of current cell to points that are incide it:
            ind = np.array(ind, dtype=bool)
            val[ind,:] = cellVal[i]
            
        return val
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            