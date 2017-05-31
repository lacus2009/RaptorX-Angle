
############################ Real value prediction ##########################
#                                                                           #
#                                                                           #
#############################################################################

### input: targetFile, which is the fasta file for the target sequence
### output:predicted angles for each residue in the sequence

###############################################################################
### Please specify your own target list and fasta_dir
if [ $# != 1 ]; then
  echo "Usage: ./PredAngles4SingleProtein.sh targetFile"
  exit 1
fi

targetFile=$1
target=${targetFile:(-11):5}
# Please setup the feature file path here for prediction
feat_file=$work_dir/out1/feat/$target.feat
id=1 # gpu ID

work_dir=$(pwd)
model_dir=$work_dir/models
sources=$work_dir/angle_prediction

k_cents_file=$model_dir/k_cents_vec
k_vars_file=$model_dir/k_vars

# You can change the output directory for angle predictions here
out_dir=$work_dir/out2
if [ ! -d "$out_dir" ]; then
  mkdir $out_dir
fi
#################### Generate marginal probability matrix #####################
ResNet_Pred=$sources/ResNet_Pred_single.py
prob_dir=$out_dir/probs
if [ ! -d "$prob_dir" ]; then
  mkdir $prob_dir
fi
MAP_list=$out_dir/$target.list
if [ -f "$MAP_list" ]; then
 rm -f $MAP_list
 touch "$MAP_list"
fi
for n_layers in {5,10,20,30,40,50}; 
do
  out_path=$prob_dir/$target.prob$n_layers
  echo $out_path >>$MAP_list
  log=$out_dir/log$n_layers
  THEANO_FLAGS=mode=FAST_RUN,device=gpu$id,floatX=float32,nvcc.flags=-arch=sm_52 python $ResNet_Pred -p $feat_file -m $model_dir/model$n_layers -o $out_path >$log 2>&1
done

consensus_mean=$sources/consensus_mean.py
out_MAP_mean=$prob_dir/$target.prob_mean
python $consensus_mean $MAP_list $out_MAP_mean

##################################################################################
pred_angles=$sources/gene_pred_angles_single.py 
pred_dir=$out_dir/pred_ResNet
lr_model=$model_dir/std2err.m
if [ ! -d "$pred_dir" ]; then
  mkdir $pred_dir
fi
python $pred_angles $out_MAP_mean $k_cents_file $k_vars_file $targetFile $lr_model $pred_dir

