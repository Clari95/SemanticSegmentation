VALEPOCH=1
LR=0.0001 #with loaded model lr 0.0015, epoch 35, acc 0.46
python train.py --data_dir $1 --val_epoch $VALEPOCH --lr $LR
