from subprocess import Popen, PIPE

import pandas as pd
import time
import os

def write_log(proc, saving_loc):
    if saving_loc is None:
        return
    '''helpful for writing logs'''
    stdout = str(proc.stdout.read())
    stderr = str(proc.stderr.read())
    stdout = stdout.replace('\\n', '\n')
    stderr = stderr.replace('\\n', '\n')
    if not os.path.exists(saving_loc):
        os.makedirs(saving_loc)
    with open(os.path.join(saving_loc, "out.txt"),"w") as f:
        f.write(stdout)
    with open(os.path.join(saving_loc, "err.txt"),"w") as f:
        f.write(stderr)
    return

class ProcBucket:
    """helpful for having `num` number of subprocesses alive and computing"""
    def __init__(self, num, sleep_time):
        self.total = 0
        self.finished = 0
        self.failed = 0
        self.num = num
        self.sleep_time = sleep_time
        self.procs = {i: None for i in range(num)}
        self.saving_locs = {i: None for i in range(num)}
        self.FNULL = open(os.devnull, 'wb')
        
    def _check_free(self):
        '''Returns the index of free slot else, returns None'''
        for i in range(self.num):
            if self.procs[i] is None: # empty
                return i
            else: # pro alloted
                rc = self.procs[i].poll()
                if rc is None: # running
                    continue
                else: # finished
                    if rc == 0: # finished successfully
                        self.finished += 1
                        write_log(self.procs[i], self.saving_locs[i])
                    else: # had recived some error
                        self.failed += 1
                        write_log(self.procs[i], self.saving_locs[i])
                    self.procs[i] = None
                    self.saving_locs[i] = None
                    return i
        return None # none empty

    def _create_proc(self, cmd, writeOutput):
        '''for creating specific process which doesn't print on stdout'''
        if writeOutput:
            return Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
        else:
            return Popen(cmd.split(), stdout=self.FNULL, stderr=self.FNULL)
        
    def add_queue(self, cmd, saving_loc=None):
        '''Adds procs to queue and blocks if already we are busy'''
        writeOutput = True
        if saving_loc is None:
            print(f"not saving output of the script\n{cmd}")
            writeOutput = False

        self.total += 1
        while True:
            rtrn_str = "Running...\n" \
                        + f"Successful: {self.finished}/{self.total}\n"\
                        + f"Failed: {self.failed}/{self.total}\n"
            ix = self._check_free()
            if ix is None: # all are busy
                time.sleep(self.sleep_time)
            else:
                self.procs[ix] = self._create_proc(cmd, writeOutput)
                self.saving_locs[ix] = saving_loc
                return rtrn_str
            
    def finalize(self):
        while True:
            for i in range(self.num):
                if self.procs[i] is not None: # filled
                    rc = self.procs[i]
                    if rc is not None:
                        write_log(self.procs[i], self.saving_locs[i])
                        self.procs[i] = None
                        self.saving_locs[i] = None
                    else:
                        time.sleep(self.sleep_time)
                else: # dont care slot empty
                    pass
            break
        return "Finished..." \
            + f"Successful: {self.finished}/{self.total}\n"\
            + f"Failed: {self.failed}/{self.total}\n"
            
