from ProcBucket import ProcBucket
from utils import getfName
import time
import argparse
import os

# argparse
parser = argparse.ArgumentParser(
    description='Caller Interpolator, Takes in datafile to run stuff ')
parser.add_argument(
    'reg', metavar='reg', type=str,
    help='Regressors to use xgb|svr|knn|las|gp'
)
parser.add_argument(
    'jobs', metavar='INT', type=int,
    help="Number of parallel jobs to run"
)
parser.add_argument(
    'stime', metavar='INT', type=int,
    help="Seconds to sleep before polling the ProcBucket"
)
parser.add_argument(
    'datafile', metavar='PATH', type=str,
    help="data csv"
)


args = parser.parse_args()
fname = getfName(args.datafile)
print ('fname:', fname)

# timer start
print ("Start Timer.")
start = time.time()

proc_bucket = ProcBucket(args.jobs, args.stime)
for lastKDays in [10, 20, 30, 50, 100, 200]:
    for stepSize in [2]:#, 5, 10, 30]: # this is fixed for now
        store_path = f"./results/{fname}/{args.reg}/{lastKDays}/{stepSize}"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        cmd = f"python svr_called.py --reg {args.reg} --stepSize {stepSize} --lastKDays {lastKDays} --datafile {args.datafile} -s"
        return_string = proc_bucket.add_queue(cmd, saving_loc=store_path)
        print ()
        print (return_string) # prints the status
return_string = proc_bucket.finalize()
print(return_string) # prints final status

end = time.time()
print("Time Taken (s):", end - start)
print()
print("Done.")
