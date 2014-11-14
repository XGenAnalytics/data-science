import pandas as pd
import numpy as np
import matplotlib
from pandas import *
from numpy import *
from matplotlib import *
%pylab inline

def xga_eda(data):
    print("Snapshot of DataFrame\n")
    df = read_csv(data)
    print df.head()
    
    num_col=0
    for col in df:
        num_col+=1
        
    print "\n------------------------------------------------------------------------\n"
    print "Total Number of Columns: \n",num_col
    
    print "\n------------------------------------------------------------------------\n"
    print "Total Number of Duplicated Rows: \n", df.duplicated().sum()
    
    print "\nData Type of Columns\n" ,df.dtypes
    
    print "\n------------------------------------------------------------------------\n"
    print "Descriptive Statistics Summary: \n\n",df.describe()
    
    print "\n------------------------------------------------------------------------\n"
    print('Data Plots Summary\n')
    
    df.isnull().sum().plot(kind='bar',figsize=(12,4),title='Missing Value Summary')
    
    df.plot(subplots=True,figsize=(12,8),color='blue',title='Data Summary')
    
    df.hist(figsize=(12,8),bins=50)

    
    
