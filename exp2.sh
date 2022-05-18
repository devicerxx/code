datasets=CIFAR10 # choises: [CIFAR10, CIFAR100, TinyImagenet]
eps=8
seed=0
#device=0

model=PreActResNet18

EXP=exp2_cyclic_30_
DST=new_results/$EXP
CUDA_VISIBLE_DEVICES=0 python -u exp2.py --wandb --cyclic \
    --datasets $datasets  --randomseed $seed \
    --train_eps $eps --test_eps $eps --train_step 1 --test_step 20 \
    --train_gamma 10 --test_gamma 2 --arch=$model \
    --epochs=30  --save-dir=$DST/models --log-dir=$DST --EXP $EXP