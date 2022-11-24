#!/bin/bash -eu
echo "Preparation============================="
mkdir -p $LOCAL_CA_PATH/council.davifn.net/assets
cp $LOCAL_CA_PATH/council.davifn.net/ca/crypto/ca-cert.pem $LOCAL_CA_PATH/council.davifn.net/assets/ca-cert.pem
cp $LOCAL_CA_PATH/council.davifn.net/ca/crypto/ca-cert.pem $LOCAL_CA_PATH/council.davifn.net/assets/tls-ca-cert.pem
echo "Preparation============================="

echo "Start Council============================="
echo "Enroll Admin"
export FABRIC_CA_CLIENT_HOME=$LOCAL_CA_PATH/council.davifn.net/registers/admin1
export FABRIC_CA_CLIENT_TLS_CERTFILES=$LOCAL_CA_PATH/council.davifn.net/assets/ca-cert.pem
export FABRIC_CA_CLIENT_MSPDIR=msp
fabric-ca-client enroll -d -u https://admin1:admin1@council.davifn.net:7050
# 加入通道时会用到admin/msp，其下必须要有admincers
mkdir -p $LOCAL_CA_PATH/council.davifn.net/registers/admin1/msp/admincerts
cp $LOCAL_CA_PATH/council.davifn.net/registers/admin1/msp/signcerts/cert.pem $LOCAL_CA_PATH/council.davifn.net/registers/admin1/msp/admincerts/cert.pem

echo "Enroll Orderer1"
# for identity
export FABRIC_CA_CLIENT_HOME=$LOCAL_CA_PATH/council.davifn.net/registers/orderer1
export FABRIC_CA_CLIENT_TLS_CERTFILES=$LOCAL_CA_PATH/council.davifn.net/assets/ca-cert.pem
export FABRIC_CA_CLIENT_MSPDIR=msp
fabric-ca-client enroll -d -u https://orderer1:orderer1@council.davifn.net:7050
mkdir -p $LOCAL_CA_PATH/council.davifn.net/registers/orderer1/msp/admincerts
cp $LOCAL_CA_PATH/council.davifn.net/registers/admin1/msp/signcerts/cert.pem $LOCAL_CA_PATH/council.davifn.net/registers/orderer1/msp/admincerts/cert.pem
# for TLS
export FABRIC_CA_CLIENT_MSPDIR=tls-msp
export FABRIC_CA_CLIENT_TLS_CERTFILES=$LOCAL_CA_PATH/council.davifn.net/assets/tls-ca-cert.pem
fabric-ca-client enroll -d -u https://orderer1:orderer1@council.davifn.net:7050 --enrollment.profile tls --csr.hosts orderer1.council.davifn.net
cp $LOCAL_CA_PATH/council.davifn.net/registers/orderer1/tls-msp/keystore/*_sk $LOCAL_CA_PATH/council.davifn.net/registers/orderer1/tls-msp/keystore/key.pem
echo "End Council============================="