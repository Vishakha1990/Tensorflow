"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""

import tensorflow as tf
import os


tf.logging.set_verbosity(tf.logging.DEBUG)
tf.set_random_seed(1024)

N = 100000 # dimension of the matrix
d = 10 # number of splits along one dimension. Thus, we will have 100 blocks
M = int(N / d)


def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)


def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)


# Create  a new graph in TensorFlow. A graph contains operators and their
# dependencies. Think of Graph in TensorFlow as a DAG. Graph is however, a more
# expressive structure. It can contain loops, conditional execution etc.
g = tf.Graph()

with g.as_default(): # make our graph the default graph

    # in the following loop, we create operators that generate individual
    # sub-matrices as tensors. Operators and tensors are created using functions
    # like tf.random_uniform, tf.constant are automatically added to the default
    # graph.
    matrices = {}
    for i in range(0, d/2):
        for j in range(i, d):
             with tf.device("/job:worker/task:%d" % (i)):
		#print ("worker id = %d matrix i=%d j=%d",  (i), i,j)
            	#print ("worker id = %d matrix i=%d j=%d",  (i), j,i)

		matrix_name = get_block_name(i, j)
            	matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)
		if(i != j):
            		matrix_name = get_block_name(j, i)
            		matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)



    for i in range(d/2, d):
        for j in range(i, d):
             with tf.device("/job:worker/task:%d" % (d-i-1)):
                #print ("worker id = %d matrix i=%d j=%d",  (d-i-1), i,j)
	        #print ("worker id = %d matrix i=%d j=%d",  (d-i-1), j,i)

                matrix_name = get_block_name(i, j)
                matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)
		if(i != j):
                	matrix_name = get_block_name(j, i)
                	matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)


    # In order the

    # In this loop, we create 100 "matmul" operators that does matrix
    # multiplication. Each "matmul" operator, takes as input two tensors as input.
    # we also create 100 "trace" operators, that takes the output of "matmul" an
    # computes the trace of the martix. Tensorflow defines a trace function;
    # however, when you observe the graph using "tensorboard" you will see that the
    # trace operator is actually implements as multiple small operators.
    intermediate_traces = {}
    for i in range(0, d/2):
        for j in range(i, d):
            with tf.device("/job:worker/task:%d" % (i)):
        	A = matrices[get_block_name(i, j)]
            	B = matrices[get_block_name(j, i)]
            	intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))




    for i in range(d/2,d):
        for j in range(i, d):
            with tf.device("/job:worker/task:%d" % (d-i-1)):
                A = matrices[get_block_name(i, j)]
                B = matrices[get_block_name(j, i)]
                intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))



    # here, we add a "add_n" operator that takes output of the "trace" operators as
    # input and produces the "retval" output tensor.
 
    with tf.device("/job:worker/task:0"):
    	retval = tf.add_n(intermediate_traces.values())



# Here, we create session. A session is required to run a computation
# represented as a graph.
    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-32-2:2222", config=config) as sess:
    #sess = tf.Session(graph=g) # create a session used to run computations on graph
    	output = sess.run(retval) # executes all necessary operations to find value of retval tensor
    
    # Summary writer is used to write the summary of execution including graph
    # structure into a log directory. By pointing "tensorboard" to this directory,
    # we will be able to graphically view the graph.
    	tf.train.SummaryWriter("%s/example_multiple" % (os.environ.get("TF_LOG_DIR")), sess.graph)
    
	sess.close()

	print "Trace of the big matrix is = ", output
