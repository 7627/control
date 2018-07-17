import numpy as np
import tensorflow as tf

freq=50.0;

#control network
st = np.matrix( np.random.rand(12,1) ); #layer 1
st_d = np.matrix(np.random.rand(12,1) );
W1 = np.matrix(np.random.rand(200,12) );
h1 = np.matrix(np.random.rand(200,1) );  #layer 2
W2 = np.matrix(np.random.rand(4,200) )
ut = np.matrix(np.random.rand(4,1) )  #layer 3

#model
X = np.vstack((ut,st[3:12,:]));  #layer 4
A1 = np.matrix(np.random.rand(100,13) )
h2 = np.matrix(np.random.rand(100,1)  )#layer 5
A2 = np.matrix(np.random.rand(3,100) )
at = np.matrix(np.random.rand(3,1) ) #layer 6_1
alpha_t = np.matrix(np.random.rand(3,1) ) #layer 6_2

#errors in layers
# del_6  = (3,1)
del_6 = at*(1/freq) + st[3:6]- st_d[3:6]; #check again

# del_5 = (100,1)
del_5 = np.multiply(np.matmul(A2.T,del_6), np.multiply(h2, (1-h2)) );

#del_4 = (13,1)
del_4 = np.matmul(A1.T,del_5);

#del_3 = (4,1)
del_3 = del_4[0:4,:];

#del_2 = (200,1)
del_2 = np.multiply(np.matmul(W2.T,del_3), np.multiply(h1, (1-h1)) );


#graidents of weights
#W1_gradient
D_W2 = np.matmul(del_3, h1.T)

#W2_gradient
D_W1 = np.matmul(del_2, st.T)Z()
