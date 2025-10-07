

IMAGE=$1
NUM_LOOPS=$2

for ((i=0; i < $NUM_LOOPS; i++)) do
    docker run --rm -d "$IMAGE"
done