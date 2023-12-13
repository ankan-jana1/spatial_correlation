#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:17:35 2022

@author: ankan_jana
"""

import pandas as pd
from pcraster import *
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
def maps_to_dataframe(maps, columns, MV = np.nan):

    """Convert a set of maps to flattened arrays as columns.

   

    Input:

        maps: list of PCRaster maps

        columns: list of strings with column names

        MV: missing values, defaults to numpy nan.

   

    Returns a pandas dataframe with missing values in any column removed.

    """

    data = OrderedDict()

    for name, pcr_map in zip(columns, maps):

        data[name] = pcraster.pcr2numpy(pcr_map, MV).flatten()

   

    return pd.DataFrame(data).dropna()

maps= albedo, lst
columns= ("albedo", "lst")
sample = maps_to_dataframe(maps, columns, MV = np.nan)


#bins=0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
#binslebels= 0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.4,0.5,0.6,0.7,0.8,0.9
bins= tuple(np.arange(0.08, 1, 0.01))
binslebels=tuple(np.arange(0.08, 0.99, 0.01))
sample['binrange']=pd.cut(sample.albedo,bins,labels=binslebels)
sample1=sample.groupby('binrange')

#for binrange, binrange_df in sample1:
#    print(binrange)
#    print(binrange_df)
    
#sample1.get_group(0.09)
maxsample=sample1.max()
maxsample=maxsample.dropna()
minsample=sample1.min()
minsample=minsample.dropna()

maxsample.plot.scatter(x='albedo', y='lst', title= "Scatter plot between two variables albedo and lst")
pmax=np.polyfit(maxsample.albedo.values,maxsample.lst.values, 1 )
print("regrassion:lst=%f albedo+%f" %tuple(pmax))
ppmax=np.poly1d(pmax)
plt.plot(maxsample.albedo.values, ppmax(maxsample.albedo.values), "r-o")
plt.show()
aH=pmax[1]
bH=pmax[0]
print(aH, bH)
minsample.plot.scatter(x='albedo', y='lst', title= "Scatter plot between two variables albedo and lst")
pmin=np.polyfit(minsample.albedo.values,minsample.lst.values, 1 )
print("regrassion:lst=%f albedo+%f" %tuple(pmin))
aLE=pmin[1]
bLE=pmin[0]
print(aLE, bLE)


ax = sample.plot(kind='scatter', x='albedo', y='lst', color='Gray',title='Scatter plot between two variables albedo and lst' );
maxsample.plot(kind ='scatter', x='albedo', y='lst', color='DarkRed', title= "Scatter plot between two variables albedo and lst", ax=ax);
pmax=np.polyfit(maxsample.albedo.values,maxsample.lst.values, 1 )
print("regrassion:lst=%f albedo+%f" %tuple(pmax))
ppmax=np.poly1d(pmax)

plt.plot(maxsample.albedo.values, ppmax(maxsample.albedo.values), "r-")
minsample.plot(kind='scatter', x='albedo', y='lst', color='DarkGreen', title='Scatter plot between two variables albedo and lst', ax=ax);
pmin=np.polyfit(minsample.albedo.values,minsample.lst.values, 1 )
print("regrassion:lst=%f albedo+%f" %tuple(pmin))
ppmin=np.poly1d(pmin)
plt.plot(minsample.albedo.values, ppmin(minsample.albedo.values), color="green")

