################# Feature Generation for a target list ################
#                                                                     #
#                                                                     #
#######################################################################

# Please specify your own target list and fasta_dir
if [ $# != 2 ]; then
  echo "Usage: ./GenFeatures4Batch.sh targetList fastaDir"
  exit 1
fi

target_list=$1
fasta_dir=$2

GenFeat4SingleProtein=$(pwd)/GenFeatures4SingleProtein.sh
for target in `cat $target_list`;
do
   targetFile=$fasta_dir/$target.fasta
   $GenFeat4SingleProtein $targetFile
done
