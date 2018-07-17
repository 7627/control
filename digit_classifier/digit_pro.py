with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        for step in range(420):
            left = step*100;
            right = min(left + 100, 42000);
            tx = train_X[left:right];
            ty = train_y[left:right];
            optimizer.run(feed_dict={X: tx, y: ty})
        
