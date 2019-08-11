for filename in $( ls $1*.sh );
do
  sbatch $filename;
done