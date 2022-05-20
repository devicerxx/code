datasets=CIFAR10 # choises: [CIFAR10, CIFAR100, TinyImagenet]
eps=8
seed=0
#device=0

model=PreActResNet18

EXP=Flooding_exp_stand_
DST=new_results/$EXP
CUDA_VISIBLE_DEVICES=0 python -u std.py \
    --datasets $datasets  --randomseed $seed --arch=$model \
    --epochs=100  --save-dir=$DST/models --log-dir=$DST --EXP $EXP