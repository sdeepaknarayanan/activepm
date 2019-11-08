import sys
import argparse

sys.path.append("../../exp1/src")

from ProcBucket import ProcBucket

parser = argparse.ArgumentParser(
    description='Helper script for running AL jobs ')
parser.add_argument(
    'kout_start', metavar='INT', type=int,
    help = "kout_start"
)
parser.add_argument(
    'kout_end', metavar='INT', type=int,
    help = "kout_end"
)
parser.add_argument(
    'kin_start', metavar='INT', type=int,
    help = "kin_start"
)
parser.add_argument(
    'kin_end', metavar='INT', type=int,
    help = "kin_end"
)
parser.add_argument(
    '--jobs', metavar='INT', type=int, default=10,
    help='number of parallel scripts to run'
)
parser.add_argument(
    '--stime', metavar='INT', type=int, default=50,
    help='Poll time.'
)

args = parser.parse_args()
pb = ProcBucket(args.jobs, args.stime)

counter = 0
for i in range(args.kout_start, args.kout_end+1):
	for j in range(args.kin_start, args.kin_end+1):
		cmd = f"bash ./scripts/{i}{j}.sh"
		print (cmd)
		st = pb.add_queue(cmd, saving_loc=f"outfiles_{i}")
		print (st)

st = pb.finalize()
print (st)
