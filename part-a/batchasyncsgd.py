import tensorflow as tf
import os
import sys

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

# number of features in the criteo dataset after one-hot encoding
num_features = 33762578


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("BatchAsyncLogfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()




g = tf.Graph()
dir = "/home/ubuntu/criteo-tfr"
with g.as_default():

    filename_queue0 = tf.train.string_input_producer([
       dir + "/tfrecords00",
       dir + "/tfrecords01",
       dir + "/tfrecords02",
       dir + "/tfrecords03",
       dir + "/tfrecords04",
    ], num_epochs=None)

    filename_queue1 = tf.train.string_input_producer([
        dir + "/tfrecords05",
        dir + "/tfrecords06",
        dir + "/tfrecords07",
        dir + "/tfrecords08",
        dir + "/tfrecords09",
    ], num_epochs=None)

    filename_queue2 = tf.train.string_input_producer([
        dir + "/tfrecords10",
        dir + "/tfrecords11",
        dir + "/tfrecords12",
        dir + "/tfrecords13",
        dir + "/tfrecords14",
    ], num_epochs=None)

    filename_queue3 = tf.train.string_input_producer([
        dir + "/tfrecords15",
        dir + "/tfrecords16",
        dir + "/tfrecords17",
        dir + "/tfrecords18",
        dir + "/tfrecords19",
    ], num_epochs=None)

    filename_queue4 = tf.train.string_input_producer([
        dir + "/tfrecords20",
        dir + "/tfrecords21",
    ], num_epochs=None)


    
    def getfileq( index ):
       if (index == 0):
       	    return filename_queue0
       elif (index == 1):
    	    return filename_queue1
       elif (index == 2):
            return filename_queue2
       elif (index == 3):
            return filename_queue3
       elif (index == 4):
            return filename_queue4


    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([num_features, 1], dtype=tf.float32), name="model")
 
    ############################## TRAINING STARTS ##############################
    # creating 5 reader operators to be placed on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        # TFRecordReader creates an operator in the graph that reads data from queue
        reader = tf.TFRecordReader(name="operator_%d" % FLAGS.task_index)

        # Include a read operator with the filenae queue to use. The output is a string
        # Tensor called serialized_example
        batch_size = 10 ;
        _, serialized_example = reader.read_up_to(getfileq(FLAGS.task_index),batch_size)

        features = tf.parse_example(serialized_example,
                                           features={
                                            'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                            'index' : tf.VarLenFeature(dtype=tf.int64),
                                            'value' : tf.VarLenFeature(dtype=tf.float32),
                                           }
                                          )
    
        label = features['label']
        index = features['index']
        value = features['value']
    
        label_split = tf.split(0,batch_size,label)
        index_split = tf.sparse_split(0,batch_size,index)
        value_split = tf.sparse_split(0,batch_size,value)


	batch_gradient = []
        for i in range(0,batch_size):
        	gathered_w = tf.gather(w,index_split[i].values)
        	exp_v = tf.expand_dims(value_split[i].values,1)
        	wdotx = tf.matmul(tf.transpose(gathered_w),exp_v)
        	y = tf.cast(label_split[i],tf.float32)
        	q = tf.sigmoid(wdotx*y) - 1
        	k = q*y
        	intermediateGradient = tf.mul(k,exp_v)
        	local_gradient = tf.mul(intermediateGradient, 0.01)
        	batch_gradient.append([tf.mul(local_gradient,0.01),index_split[i].values])
		
    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
	    #new_w = tf.expand_dims(w,1)
    	for i in range(0, batch_size):
            w = tf.scatter_sub(w,batch_gradient[i][1],batch_gradient[i][0])
    ################### TRAINING ENDS #############################################


    ################### TEST STARTS #############################################
        with tf.device("/job:worker/task:%d" % 0):
            # TFRecordReader creates an operator in the graph that reads data from queue
            reader = tf.TFRecordReader(name="operator_%d" % 0)

            test_filename_queue = tf.train.string_input_producer([
                dir + "/tfrecords22",
            ], num_epochs=None)

	    
            _, serialized_example = reader.read(test_filename_queue)
            features = tf.parse_single_example(serialized_example,
                                              features={
                                               'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                               'index' : tf.VarLenFeature(dtype=tf.int64),
                                               'value' : tf.VarLenFeature(dtype=tf.float32),
                                              }
                                             )

            label = features['label']
            index = features['index']
            value = features['value']

            dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                          [num_features,],
           #                               tf.constant([33762578, 1], dtype=tf.int64),
                                          tf.sparse_tensor_to_dense(value))

            dense_feature = tf.reshape(dense_feature, [num_features,1])
            dotProduct = tf.matmul(tf.transpose(w), dense_feature)
	    y = tf.cast(label,tf.float32)
	    error = tf.not_equal(tf.sign(dotProduct),y)
  
		


    ################### TEST ENDS #############################################

      	with tf.Session("grpc://vm-32-%d:2222" % (FLAGS.task_index+1)) as sess:
	        # only one client initializes the variable
           if FLAGS.task_index == 0:
               sess.run(tf.initialize_all_variables())
   
      	  # start queue runners
           coord = tf.train.Coordinator()
           threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	   num_examples= 1000;
           for i in range(0, 10000):
	    sess.run(w)
	    print i
	    if((i%100)==0):
	    	e= 0 ;
	    	for j in range(0,num_examples):
	     		output = sess.run(error)
			e = e + int(output)
	        err_rate = float(e)/(num_examples)
	    	print ("error is" , float(err_rate))	
	   print w.eval()
        # tried to stop the queue runners
           coord.request_stop()
           coord.join(threads)
           sess.close()


