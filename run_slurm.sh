#!/bin/bash

if ! OPTIONS=$(getopt -o a:,c:,m:,g: -l args:,cpu:,mem:,gpu:,exc: -- "$@")
then
    exit 1
fi

eval set -- "$OPTIONS"

while [ $# -gt 0 ]
do
    case $1 in
    -a|--args) $ARGS=${2} ; shift ;;
    -c|--cpu) CPU=${2} ; shift ;;
    -m|--mem) MEM=${2} ; shift ;;
    -g|--gpu) GPU=${2} ; shift ;;
    --exc) EXCLUDE=${2} ; shift ;;
    (--) shift; break;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
    (*) break;;
    esac
    shift
done

if [ -z "$ARGS" ]; then
    ARGS=""
fi

if [ -z "$CPU" ]; then
    CPU=2
fi

if [ -z "$MEM" ]; then
    MEM=20
fi

if [ -z "$GPU" ]; then
    GPU=8
fi

SEED=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff ))
DATA="imagenet"
BS=16

JOB=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=emcom                # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=$CPU            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem="$MEM"G                   # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=shard:$GPU               # number of gpus per node
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --time=24:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=fabiovital@tecnico.ulisboa.pt
#SBATCH --exclude=$EXCLUDE

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.46
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset XLA_FLAGS

python src/noisy_emcom/main.py \
    --config=src/noisy_emcom/configs/lg_rlrl_config.py:$DATA \
    --config.random_seed=$(( ((RANDOM<<30) | (RANDOM<<15) | RANDOM) & 0x7fffffff )) \
    --config.batch_size=$BS \

EOF
)
