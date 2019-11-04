# TODO: create shell script for running the testing code of the baseline model
wget https://www.dropbox.com/s/zjzwplw4wr9mvx2/model_best_best.pth.tar?dl=1
RESUME='model_best_best.pth.tar?dl=1'
python test.py --resume $RESUME --data_dir $1 --save_dir $2

