import sys
import os

sys.path.append("../../")
from exp1.src.ProcBucket import ProcBucket

# pb = ProcBucket(10, 3)

for k in [1]:
    for i in range(6):
        for j in range(5):
            if not os.path.exists(f"knn_al_results/{i}_{j}_{k}"):
                cmd = f"python called_knn.py --kout {i} --kin {j} --train_days 30 --act {k}"
                ste = pb.add_queue(cmd, saving_loc=f"knn_al_results/{i}_{j}_{k}")
                print (ste)
#                 print (cmd)
            # break
        # break
    # break
ste = pb.finalize()
print (ste)