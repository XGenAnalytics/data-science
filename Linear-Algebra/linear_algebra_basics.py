
# coding: utf-8

# In[1]:

######################### Vectors, Spaces & Matrices Tutorial ###########################


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import arange
from numpy import meshgrid
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15.0, 5.0)


# In[3]:

# Multiplication of Matrix


# In[4]:

A=matrix([[3,2],[6,5]])
B= matrix([[1,2],[4,5]])
print ("Multiplicaton of Matrices\n")
A*B


# In[5]:

# Visualizing 2 dimensional linear equations


# In[6]:

x=np.array(range(0,20))
y=eval('(6-x)')
a=np.array(range(0,20))
b=eval('a+2')
plot(x,y)
plot(a,b)
pylab.xlim([-2,25])
pylab.ylim([-20,25])


# In[7]:

## Plotting 3d linear equations on plane


# In[8]:

a  = np.array([0,1,1])
b  = np.array([1,-2,0])
c  = np.array([1,0,1])
n1 = np.array([7,1,1])
n2 = np.array([1,-3,9])
n3 = np.array([-5,6,9])

d1 = -np.dot(a,n1)
d2 = -np.dot(b,n2)
d3 = -np.dot(c,n3)
xx, yy = np.meshgrid(range(30), range(30))

z1 = (-n1[0]*xx - n1[1]*yy - d1)*1./n1[2]
z2 = (-n2[0]*xx - n2[1]*yy - d2)*1./n2[2]
z3 = (-n3[0]*xx - n3[1]*yy - d3)*1./n3[2]

plt3 = plt.figure().gca(projection='3d')
plt3.plot_surface(z1,yy,xx, color='red')
plt3.plot_surface(z2,yy,xx, color='green')
plt3.plot_surface(z3,yy,xx, color='yellow')


# In[9]:

# Vector addition


# In[10]:

print "Vector Addition\n"
matrix([3,6,4]) + matrix([1,2,3])


# In[11]:

matrix([1,2,3])*2


# In[12]:

matrix([[3],[6],[4]]) + matrix([[1],[2],[3]])


# In[13]:

# Solving 3d linear equations


# In[14]:

a = np.matrix([[2,3,4],[1,-2,8],[1,2,3]])
b = numpy.matrix([[1],[2],[3]])
scipy.linalg.solve(a, b)


# In[15]:

# Solving quadratic equations


# In[16]:

a = 1
b = 22
c = 1

d = b**2-4*a*c 

if d == 0:
    x = (-b+math.sqrt(b**2-4*a*c))/2*a
    print x
else:
    x1 = (-b+math.sqrt((b**2)-(4*(a*c))))/(2*a)
    x2 = (-b-math.sqrt((b**2)-(4*(a*c))))/(2*a)
    print ("two solutions: ", x1, " or", x2)


# In[17]:

# Identity Matrix


# In[18]:

matrix([[1,0,0],[0,1,0],[0,0,1]])*matrix([[5],[-2],[9]])


# In[19]:

matrix([[1,0,0],[0,1,0],[0,0,1]])*10


# In[20]:

# AInverse * A = Identity_Matrix


# In[21]:

A=matrix([[1,3,0],[4,1,1],[2,3,1]])
B=matrix([[2,3,4],[4,4,6],[3,8,4]])
I=matrix([[1,0,0],[0,1,0],[0,0,1]])


# In[22]:

A.I


# In[23]:

A*A.I==I


# In[24]:

# Distributive Property (A*B).I = B.I*A.I


# In[25]:

(A*B).I


# In[26]:

B.I*A.I


# In[27]:

# Transpose of Matrix


# In[28]:

A.T


# In[29]:

# Transpose Matrix Distributive Property(AB)T = BT * AT


# In[30]:

(A*B).T


# In[31]:

B.T*A.T


# In[32]:

A.I


# In[33]:

# Displaying Vectors (X,Y, Addition, Multiplication)


# In[34]:

X = np.array([2,1])
Y = np.array([-4,4])
SUM = X+Y
MUL = X*Y
ax = plt.gca()
ax.quiver(X,Y,angles='xy',scale_units='xy',scale=1,color='green',label='x')
ax.quiver(SUM,MUL,angles='xy',scale_units='xy',scale=1,color='red')
ax.set_xlim([-5,10])
ax.set_ylim([-10,10])


# In[35]:

# Displaying Multiplication of Vector by Scalar


# In[36]:

X = np.array([2,1])
MUL = 3*X
ax = plt.gca()
ax.quiver(X,MUL,angles='xy',scale_units='xy',scale=1,animated=True,label='x',color='green')
ax.set_xlim([-2,5])
ax.set_ylim([-2,10])


# In[37]:

# Unit Vectors


# In[38]:

v1=np.array([1,2])
v2=np.array([0,3])
3*v1-2*v2


# In[39]:

# Solving linear equations using Scipy


# In[40]:

import scipy.linalg
a = numpy.matrix([[1,2,-1],[-1,1,0],[2,3,2]])
b = numpy.matrix([[2],[3],[4]])
print scipy.linalg.solve(a, b)


# In[41]:

#Vector dot product 


# In[42]:

a=[2,5]
b=[7,1]
np.dot(a,b)


# In[43]:

a=[[2,5],[3,4]]
b=[[7,1],[5,6]]
np.dot(a,b)


# In[44]:

#Vector Length


# In[45]:

x = np.array([3,4])
np.linalg.norm(x)


# In[46]:

sqrt(np.dot(x,x))==np.linalg.norm(x)


# In[47]:

#Vector Dot Product Properties


# In[48]:

# Commutative Property
a=[2,5,4]
b=[7,1,6]
np.dot(a,b)==np.dot(b,a)


# In[49]:

#Associative
a=np.array([4,10,8])
c=np.array([2,5,4])
b=np.array([7,1,6])
np.dot(a,b)==2*np.dot(c,b)


# In[50]:

# Cauchy–Schwarz inequality


# In[51]:

a=np.array([4,10,8])
b=np.array([7,1,6])
np.dot(a,b) <= (np.linalg.norm(a))*(np.linalg.norm(b))


# In[52]:

# Angle between Vectors


# In[53]:

a=np.array([4,10,8])
b=np.array([7,1,6])
cosang = np.dot(a,b)
sinang = norm(np.cross(a,b))
np.arctan2(sinang, cosang)


# In[54]:

np.angle(1+1j, deg=True)


# In[55]:

#Vector Cross Product
a = np.array([1, -7, 1])
b = np.array([5, 2, 4])
np.cross(a,b)


# In[56]:

# Vector Triple Product
import numpy as np
a = np.array([1, -7, 1])
b = np.array([5, 2, 4])
c = np.array([3, 0, 4])
np.cross(np.cross(a,b),c)


# In[57]:

# Matrix Product


# In[58]:

a = np.mat([[-3, 0,3,2],[1,7,-1,9]])
b = np.mat([[2],[-3],[4],[-1]])
print a*b


# In[59]:

# Null Vector Space 


# In[60]:

def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)
a = np.mat([[1,1,1,1],[1,2,3,4],[4,3,2,1]])
a*null(a)


# In[61]:

# Matrix Distributive Property


# In[62]:

a=np.matrix([[2,-1],[3,4]])
b=np.matrix([[2],[3]])
4*(a+b)==4*a+4*b


# In[63]:

# Matrix Multiplication with identity Matrix


# In[64]:

a=np.matrix([[2,-1],[3,4]])
I=np.matrix([[1,0],[0,1]])
a*I==a


# In[65]:

# Inverse of matrix


# In[66]:

a=np.matrix([[2,-1],[3,4]])
a.I


# In[67]:

# Determinants
a=np.matrix([[1,2,4],[2,-1,3],[4,0,1]])
np.linalg.det(a)


# In[68]:

# Determinant multiplied by Scalar
a=np.matrix([[1,2,4],[2,-1,3],[4,0,1]])
5*np.linalg.det(a)


# In[69]:

# Solving 4x4 Determinant
a=np.matrix([[1,2,2,1],[1,2,4,2],[2,7,5,2],[-1,4,-6,3]])
np.linalg.det(a)


# In[70]:

# Determinant of Transpose
a=np.matrix([[1,2,4],[2,-1,3],[4,0,1]])
np.linalg.det(a.T)


# In[71]:

# Transpose of a vector
a=np.array([[2],[3],[3],5])
a.T


# In[72]:

# Rank of a Matrix
a=np.matrix([[1,2,4,5],[2,-1,3,6],[4,0,1,7],[5,2,8,5]])
np.linalg.matrix_rank(a)

