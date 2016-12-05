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
        self.log = open("AsyncLogfile.log", "a")

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
        _, serialized_example = reader.read(getfileq(FLAGS.task_index))

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

        ## This is sparse implementation without any batching 
        gathered_w = tf.gather(w,index.values)
        exp_v = tf.expand_dims(value.values,1)
        z = tf.matmul(tf.transpose(gathered_w),exp_v)
        y = tf.cast(label,tf.float32)
        q = tf.sigmoid(y*z) - 1
        k = q*y*0.01
        local_gradient = tf.mul(k,exp_v)
    
    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        w = tf.scatter_sub(w,index.values,local_gradient)



		
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
           for i in xrange(0, 10000):
	    sess.run(w)
	    if((i%1000)==0):
	   	e= 0 ;
	    	for j in xrange(0,num_examples):
	     		output = sess.run(error)
			e = e + int(output)
	        err_rate = float(e)/(num_examples)
	        print ("error is" , float(err_rate))	
	   print w.eval()
        # tried to stop the queue runners
           coord.request_stop()
           coord.join(threads)
           sess.close()


