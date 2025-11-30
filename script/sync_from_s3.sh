echo "File in s3:" 
s3cmd ls s3://b4schnei/pokeagent/internet_data/ | wc -l
echo "File in local:" 
ls -l .cache/pokeagent/internet_data | wc -l
python ./s3_utils/s3_sync.py .cache pokeagent/internet_data --bucket b4schnei --mode sync 