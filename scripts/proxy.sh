# ip=10.61.2.90
ip=10.69.22.203
port=1082
export http_proxy=http://${ip}:${port}
export https_proxy=http://${ip}:${port}
source ~/.bashrc
curl -I https://www.google.com