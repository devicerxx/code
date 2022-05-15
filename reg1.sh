datasets=CIFAR10 # choises: [CIFAR10, CIFAR100, TinyImagenet]
eps=8
seed=0
#device=0

model=PreActResNet18

EXP=reg1_
DST=new_results/$EXP
CUDA_VISIBLE_DEVICES=1 python -u reg1.py --wandb \
    --datasets $datasets  --randomseed $seed \
    --train_eps $eps --test_eps $eps --train_step 1 --test_step 20 --batch-size 64 --wd 5e-4 \
    --train_gamma 10 --test_gamma 2 --arch=$model \
    --epochs=100  --save-dir=$DST/models --log-dir=$DST --EXP $EXP