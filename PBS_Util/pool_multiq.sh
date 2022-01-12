
#PBS -lselect=1:ncpus=16:ompthreads=16:mem=20gb
#PBS -l walltime=16:00:00

# Load modules for any applications

module load anaconda3/personal
module load cuda

# Change to the submission directory
cd $PBS_O_WORKDIR

# Run program
# The program won't automatically use more than 1 core
# It has to be written with parallel capability


source activate py38
echo "Job is ${job} with episodes ${episodes} and trials ${trials}"
echo "Characteristics are ${slr} ${dlr} ${ada} ${aser} ${lr} ${R_D}"

if [ -z "${lr}" ]
then
  echo "lr is empty"
else
  slr="-l ${lr}"
fi

if [ -z "${dlr}" ]
then
  echo "dlr is empty"
else
  dlr="-d ${dlr}"
fi

if [ -z "${ada}" ]
then
  echo "ada is empty"
else
  ada="-m ${ada}"
fi

if [ -z "${aser}" ]
then
  echo "aser is empty"
else
  aser="-n ${aser}"
fi

if [ -z "${tda}" ]
then
  echo "tda is empty"
else
  tda="-j ${tda}"
fi

if [ -z "${R_D}" ]
then
  echo "R_D is empty"
else
  R_D="-r ${R_D}"
fi

if [ -z "${etSero}" ]
then
  echo "etSero is empty"
else
  etSero="-y ${etSero}"
fi

echo "python comp_neuromod_MWM_online.py -o ${job}  -e ${episodes}  -s -t ${trials} $lr $dlr $ada $aser $R_D $etSero"
python comp_neuromod_MWM_online.py -o ${job} -e ${episodes} -s -t ${trials} $lr $dlr $ada $aser $R_D $etSero

# The number of cores assigned to the job is available
# In the environment variable NCPUS or OMP_NUM_THREADS

a.out
