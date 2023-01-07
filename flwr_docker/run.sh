#!/usr/bin/env bash

help() 
{
    echo 'bash run.bash or chmod +x run.bash && ./run.bash'
    echo 'parameters:'
    echo '-h, server ip address, eg. -h 127.0.0.1'
    echo '-p, server port, eg. -p 12345'
    echo '-v, data volume, could be volume name or local path, eg. -v /data'
    echo '-i, image name, eg. -i image_name'
    echo '-c, container name, eg. -c container_name'
    echo '-H, help, eg. -H'
    # shellcheck disable=SC2242
    exit -1
}

if [ $# -le 1 ]; 
then
    help
fi

while getopts "h:p:v:i:c:H" arg
do
    case $arg in
        h) host=$OPTARG;;
        p) port=$OPTARG;; 
        v) volume=$OPTARG;;
        i) image=$OPTARG;;
        c) container=$OPTARG;;
        H) help;;
        ?) help;;
    esac
done

echo $host
echo $port

# shellcheck disable=SC2236
if [[ ! -n "$host" ]]
then
    echo 'Must be a server ip address'
    # shellcheck disable=SC2242
    exit -1
fi

# shellcheck disable=SC2236
if [[ ! -n "$port" ]]
then
    echo 'Must be a port'
    # shellcheck disable=SC2242
    exit -1
fi

# shellcheck disable=SC2236
if [[ ! -n "$image" ]]
then
    image='flwr_pytorch_client'
fi

# shellcheck disable=SC2236
if [[ ! -n "$container" ]]
then
    container='flwr_pytorch_client'
fi

docker build --build-arg IP=$host --build-arg PORT=$port --force-rm -t $image .
# shellcheck disable=SC2236
if [[ ! -n "$volume" ]]
then
    docker run --rm  --name=$container -d $image
else
    docker run --rm  --name=$container -v $volume:/data:ro -d $image
fi
