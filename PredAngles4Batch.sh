
############################ Real value prediction ##########################
#                                                                           #
#                                                                           #
#############################################################################

### input: target_list, fasta_dir
### output:predicted angles for each protein on target list
#          Please setup the output directions in PredAngles4SingleProtein.sh (out_dir)

###############################################################################
### Please specify your own target list and fasta_dir
if [ $# != 2 ]; then
  echo "Usage: ./PredAngles4Batch.sh target_list fasta_dir"
  exit 1
fi

target_list=$1
fasta_dir=$2

PredAngles4SingleProtein=$(pwd)/PredAngles4SingleProtein.sh
for target in `cat $target_list`;
do
   targetFile=$fasta_dir/$target.fasta
   $PredAngles4SingleProtein $targetFile
done
