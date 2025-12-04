echo "File in s3:" 
s3cmd ls s3://b4schnei/pokeagent/dinov2/ | wc -l
echo "File in local:" 
ls -l .cache/pokeagent/dinov2 | wc -l
python ./s3_utils/s3_sync.py .cache/pokeagent/dinov2 pokeagent/dinov2 --bucket b4schnei --mode upload 