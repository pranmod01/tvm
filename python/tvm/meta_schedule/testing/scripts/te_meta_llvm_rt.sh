set -euxo pipefail
  
RPC_HOST=172.31.55.65
RPC_PORT=4445

RPC_KEY="c5.9xlarge"
TARGET="llvm -mcpu=skylake-avx512 -num-cores 18"

RPC_WORKERS=1
NUM_TRIALS=10000
CMD="tvm.meta_schedule.testing.tune_te_meta_schedule_rt"


run () {
    name=$1
    LOG_ID=$2
    LOG_DIR=$HOME/logs/ms-rt-llvm-t2d-$LOG_ID/
    WORK_DIR=$LOG_DIR/$name
    mkdir -p $LOG_DIR
    mkdir -p $WORK_DIR
    echo "Running workload $name @ $LOG_DIR"
    python -m $CMD                          \
        --workload "$name"                  \
        --target "$TARGET"                  \
        --rpc-host "$RPC_HOST"              \
        --rpc-port "$RPC_PORT"              \
        --rpc-key "$RPC_KEY"                \
        --rpc-workers "$RPC_WORKERS"        \
        --num-trials $NUM_TRIALS            \
        --work-dir "$WORK_DIR"              \
        2>&1 | tee -a "$WORK_DIR/$name.log"
}

process () {
    #run C1D $1
    #run C2D $1
    #run CAP $1
    #run DEP $1
    #run DIL $1
    #run GMM $1
    #run GRP $1
    run T2D $1
    #run C2d-BN-RELU $1
    #run TBG $1
    #run NRM $1
    #run SFM $1
    #run C3D $1
}


process 1
#process 2
#process 3
#process 4
#process 5
