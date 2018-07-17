# https://www.kaggle.com/c/digit-recognizer/data
#MNIST beginner
#python3
import numpy as np
import tensorflow as tf
from math import ceil

train_data = np.genfromtxt('train.csv', delimiter=',' , skip_header=1);
test_data = np.genfromtxt('test.csv', delimiter=',' , skip_header=1);
train_X = train_data[:,1:]
train_y = train_data[:,0]
test_X = test_data

input_layer = 784;
hidden_layer = 200;
output_layer = 10;
batch_size = 100;
num_epoch = 1000

temp = np.zeros([train_y.size, output_layer])
for i in range(train_y.size):
    temp[i][int(train_y[i])] = 1
#
train_y = temp;
del(temp)

X = tf.placeholder(tf.float32, shape=[None,input_layer]);
y = tf.placeholder(tf.float32, shape=[None, output_layer]);

W1 = tf.Variable(tf.truncated_normal([input_layer, hidden_layer]), name="W1");
b1 = tf.Variable(tf.truncated_normal([hidden_layer]), name="b1")
W2 = tf.Variable(tf.truncated_normal([hidden_layer, output_layer]), name="W2")
b2 = tf.Variable(tf.truncated_normal([output_layer]), name="b2")

H = tf.nn.sigmoid(tf.add(tf.matmul(X,W1), b1) )
output = tf.nn.softmax(tf.add(tf.matmul(H,W2), b2) )

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), reduction_indices=[1]));
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output) + (1-y)*tf.log(1-output), reduction_indices=[1]));

optimizer = tf.train.AdamOptimizer().minimize(cost);

init = tf.global_variables_initializer();

num_batches = int(ceil(float(train_X.shape[0])/batch_size))
print("num_batches = ", num_batches);

with tf.Session() as sess:
    sess.run(init);
    for epoch in range(num_epoch):
        running_cost = 0
        for step in range(num_batches):
            left = step*batch_size;
            right = min(left + batch_size , train_X.shape[0])
            tx = train_X[left:right]
            ty = train_y[left:right]
            _,r_c = sess.run([optimizer, cost],feed_dict={X: tx, y: ty})
            running_cost += r_c;
            # print("step= ",step, " running_cost= ",running_cost);

        epoch_cost = running_cost/num_batches
        print("epoch= ",epoch+1, " epoch_cost= ", epoch_cost);

        # y_pred = sess.run([output],feed_dict={X:train_X, y:train_y})
        if epoch%10==0:
            pred = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
            print("\naccuracy = ",sess.run(accuracy, feed_dict={X:train_X, y:train_y}),"\n")

    y_pred = tf.argmax(output,1)
    y_pred = sess.run(y_pred, feed_dict={X:test_X})
    with open('y_pred.txt','a') as file:
        for i in range(y_pred.size):
            file.write(str(i+1)+','+str(y_pred[i])+'\n')










#
