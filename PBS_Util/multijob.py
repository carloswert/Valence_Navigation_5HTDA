import subprocess
import itertools
import psutil
import time
import re
from numpy import arange

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


#gridsArgs_CWC = {'job':['job_CWC_Sero'],'trials':[1000],'episodes':[20],'ada':[1],'aser':[0.01],'lr':[0.0001],'R_D':[1]}
gridsArgs_SWC = {'job':['job_SWC_Sero'],'trials':[1000],'episodes':[20],'ada':[1],'aser':[1],'lr':[0.01]}
if gridsArgs['episodes'][0]>1000:
    gridsArgs['episodes'] = [1000]*(gridsArgs['episodes'][0]//1000)

args = dict_product(gridsArgs)
for i in enumerate(args):
    opts = [v is not None for v in [i[1].get('lr'),i[1].get('d_lr'),i[1].get('ada'),i[1].get('aser'),i[1].get('R_D'),i[1].get('tda'),i[1].get('etSero')]]
    cmd="qsub -N "+str(i[1].get('job'))+"_"+str(i[0])+" -v job="+str(i[1].get('job'))+"_"+str(i[0])+",episodes="+str(i[1].get('episodes'))+",trials="+str(i[1].get('trials'))+opts[0]*(",lr="+str(i[1].get('lr')))+opts[1]*(",dlr="+str(i[1].get('d_lr')))+opts[2]*(",ada="+str(i[1].get('ada')))+opts[3]*(",aser="+str(i[1].get('aser')))+opts[4]*(",R_D="+str(i[1].get('R_D')))+opts[5]*(",tda="+str(i[1].get('tda')))+opts[6]*(",etSero="+str(i[1].get('etSero')))+" pool_multiq.sh"
    print(cmd)
    p=subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    njobs = int(subprocess.check_output('qselect -u <username> | wc -l',shell=True).decode('utf-8'))
    #Job limit for crossvalidation
    while njobs>40:
        time.sleep(1)
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        njobs = int(subprocess.check_output('qselect -u <username> | wc -l',shell=True).decode('utf-8'))
