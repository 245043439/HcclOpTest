name=$1
image=$2
# image=vllm-ascend:dev-26.1.0.B060-A5-server-py311-openEuler24.03-lts-aarch64
# image=vllm-ascend:dev-26.1.0.B050-A5-server-py311-openEuler24.03-lts-aarch64
# image=lbt_test:vllm_0603
# image=lbt_test:vllm_0601
# image=lbt_test:vllm_0530
# image=vllm-ascend:dev-26.1.0.B030v3-A5-800I-A5-py311-openEuler24.03-lts-aarch64

docker run --runtime=runc -u root -it -d --name ${name} --net=host --privileged=true --shm-size=2g \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /root/host:/root/host  \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/usr/slog \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json  \
    -v /usr/lib64:/usr/lib64   \
    -v /lib64:/lib64 \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /home/:/home/ \
    -v /etc/hixlep:/etc/hixlep \
    -w /workspace \
    --entrypoint=/bin/bash \
    $image \
