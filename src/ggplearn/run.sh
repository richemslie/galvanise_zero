# create directory for data
mkdir -p data/$1

# move anything local there
mv mcs_$1* data/$1

# rsync from sim
rsync -vaz  rxe@sim:/home/rxe/working/ggplearn/src/ggplearn/mcs_$1* data/$1

# rsync from amazon
#rsync -vazre "ssh -i /home/rxe/.ssh/spotty.pem"   ubuntu@54.77.192.94:ggplearn/src/ggplearn/mcs_$1* data/$1
sleep 2

python process_games.py data/$1 $1 $2

mv model_nn_$1_$2.json models/
mv weights_nn_$1_$2.h5 models/
mv process_and_train__$1_$2*.log
echo "DONE"
