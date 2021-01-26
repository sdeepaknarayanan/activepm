# activepm
Codes and results for active learning for station deployment
Run all experiments from the master branch

Instructions for running the code:
* For the interpolation experiment,
    * Go to exp1/src/. Then run called.py with the arguments or caller.py with the corresponding arguments.
* For the active learning experiment,
    * Go to exp2/src/. Then run called.py with the arguments or caller.py with the corresponding arguments.
    
For experiment 2 - Active Learning - Use 

Scripts in the following format:

Here kout is outer fold, kin is inner fold, act is if active learning or not, set id to default to 0 - id is the gpu id, called1.py is the script name and train days is the number of days used for training purposes. 
Regressor names can be obtained via using the help in argparse.

```python called1.py --kout 3 --kin 0 --train_days 10 --act 0 --id 0```

For experiment 1 - interpolation - use

```python called.py --reg <reg> --stepSize 2 -lastKDays 10 --datafile <path> --totalDays 350 -s True --loc <location to store, no need to change>```
Choose regressors by using help for argparse, path as datasets (beijingscaleda and beijingscaledb for feature sets a and b).

before running experiments, kindly remove all the folders named results so that results can be regenerated. 
To obtain the mean errors, kindly take the mean of the rmse's of the generated csv's.

Please cite us if you are using our work:
```bibtex
@inproceedings{realml_20,
    title={Active Learning for Air Quality Station Deployment},
    author={S Deepak Narayanan and Apoorv Agnihotri and Nipun Batra},
    booktitle={ICML 2020 Workshop on Real World Experiment Design and Active Learning},
    year={2020}
}
```bibtex
@inproceedings{10.1145/3371158.3371208,
author = {Narayanan, S. Deepak and Agnihotri, Apoorv and Batra, Nipun},
title = {Active Learning for Air Quality Station Location Recommendation},
year = {2020},
isbn = {9781450377386},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3371158.3371208},
doi = {10.1145/3371158.3371208},
booktitle = {Proceedings of the 7th ACM IKDD CoDS and 25th COMAD},
pages = {326–327},
numpages = {2},
location = {Hyderabad, India},
series = {CoDS COMAD 2020}
}
```
  
