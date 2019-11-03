RESUME='./log/model_best.pth.tar'
python test.py --resume $RESUME --data_dir $1 --save_dir $2
