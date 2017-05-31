################# Feature Generation for a target list ################
#                                                                     #
#                                                                     #
#######################################################################

# Please specify your own target list and fasta_dir
if [ $# != 1 ]; then
  echo "Usage: ./GenFeatures4SingleProtein.sh targetFile"
  exit 1
fi

targetFile=$1
target=${targetFile:(-11):5}

work_dir=$(pwd)
feat_gene=$work_dir/feature_generation

# You can change output directory for feature generation here
out_dir=$work_dir/out1
if [ ! -d "$out_dir" ]; then
  mkdir $out_dir
fi

############ 1. prepare files
####### input:  targetFile (fasta file for the query target)
#       output: tgt file

TGT_package=$feat_gene/TGT_Package
Fast_TGT=$TGT_package/Fast_TGT.sh
tgt_dir=$out_dir/tgt_out
if [ ! -d "$tgt_dir" ]; then
  mkdir $tgt_dir
fi

cd $TGT_package
$Fast_TGT -i $targetFile -o $tgt_dir
cd $work_dir 

############ 2. generate 66feat data
####### input:  targetFile
#               tgt file
#       output: feature file (record the features for each residue in the target sequence)
gene_66feat=$feat_gene/gene_66feat_single.py
feat_dir=$out_dir/feat
if [ ! -d "$feat_dir" ]; then
  mkdir $feat_dir
fi
feat_file=$feat_dir/$target.feat
python $gene_66feat $targetFile $tgt_dir/$target.tgt $feat_file

