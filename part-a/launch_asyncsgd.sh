#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source ./tfdefs.sh
terminate_cluster
pdsh -R ssh -w ^machines "killall python"
rm out_runSYN_sync

export TF_LOG_DIR="/home/ubuntu/tf/logs"
# startserver.py has the specifications for the cluster.
start_cluster startserver.py
pdsh -R ssh -w ^machines "mkdir -p ~/assign3/results/;"
pdsh -R ssh -w ^machines "rm ~/assign3/results/dstat_res;"
sleep 10

pdsh -R ssh -w ^machines "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

pdsh -R ssh -w ^machines 'sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches";'
sleep 10

pdsh -R ssh -w ^machines 'dstat -tcmsdn 30 > ~/assign3/results/dstat_res;' &
sleep 2

echo "Executing the distributed tensorflow job"
# testdistributed.py is a client that can run jobs on the cluster.
# please read testdistributed.py to understand the steps defining a Graph and
# launch a session to run the Graph
python /home/ubuntu/grader/part-a/asyncsgd.py --task_index=0
sleep 2
python /home/ubuntu/grader/part-a/asyncsgd.py --task_index=1
python /home/ubuntu/grader/part-a/asyncsgd.py --task_index=2
python /home/ubuntu/grader/part-a/asyncsgd.py --task_index=3
python /home/ubuntu/grader/part-a/asyncsgd.py --task_index=4

# defined in tfdefs.sh to terminate the cluster
terminate_cluster

pdsh -R ssh -w ^machines "ps -ef | grep dstat | awk '{print \$2}' | xargs kill"
sleep 10

scp vm-32-2:~/assign3/results/dstat_res ~/assign3/results/dstat_res2
scp vm-32-3:~/assign3/results/dstat_res ~/assign3/results/dstat_res3
scp vm-32-4:~/assign3/results/dstat_res ~/assign3/results/dstat_res4
scp vm-32-5:~/assign3/results/dstat_res ~/assign3/results/dstat_res5


# TODO: Uncomment this for printing
#tensorboard --logdir=$TF_LOG_DIR
