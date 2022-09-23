param(
    [String]$ip,
    [Int32]$port,
    [String]$image="flwr_client",
    [String]$container="flwr_client",
    [String]$volume="",
    [Switch]$help=$false
)

function help
{
    Write-Output ".\run.ps1"
    Write-Output "parameters:"
    Write-Output "-ip, server ip address, eg. -ip 127.0.0.1"
    Write-Output "-port, server port, eg. -port 12345"
    Write-Output "-image, image name, eg. -image image_name"
    Write-Output "-container, container name, eg. -container container_name"
    Write-Output "-volume, data volume, eg. -volume /data"
    Write-Output "-help, help doc, eg. -help:$true or -help:$false"
    exit
}

if ($PSBoundParameters.Count -eq 0)
{
    help
}

if ($help)
{
    help
}

if ($ip -eq "")
{
    Write-Output "Must be a server ip address"
    exit
}

if ($port -eq 0)
{
    Write-Output "Must be a server port"
    exit
}

docker.exe build --build-arg IP=$ip --build-arg PORT=$port --force-rm -t $image .
if ($volume -eq "")
{
    docker run --name=$container -d $image
}
else
{
    Write-Output $volume
    docker run --name=$container -v ${volume}:/data:ro -d $image
}