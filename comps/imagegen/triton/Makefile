build:
	docker build -t base-installer-ubuntu22.04:1.15.1-15 -f Dockerfile.base .
	docker build -t base -f Dockerfile.ubuntu .
	docker build --progress=plain -t ohio-stability-triton -f Dockerfile .

DEVICES?=3
run:
	docker run \
		-e HABANA_VISIBLE_DEVICES=$(DEVICES) \
		--runtime habana \
		--shm-size=1g \
		--ulimit memlock=-1 \
		-p 18000:8000 \
		-p 18001:8001 \
		-p 18002:8002 \
		--ulimit stack=67108864 \
		-v /home/sdp/.cache/huggingface/hub:/root/.cache/huggingface/hub \
		-ti test

test:
	docker run --runtime runc -ti --net host nvcr.io/nvidia/tritonserver:24.04-py3-sdk \
		bash -c "git clone https://github.com/triton-inference-server/python_backend -b r24.04 && python3 python_backend/examples/add_sub/client.py"
