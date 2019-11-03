# TODO: create shell script for running the testing code of the baseline model
wget https://www.dropbox.com/s/skymm88jdu1veaa/model_best.pth.tar?dl=1

RESUME='model_best.pth.tar'
python test.py --resume $RESUME --data_dir $1 --save_dir $2

