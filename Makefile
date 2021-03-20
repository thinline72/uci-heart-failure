IMAGE_NAME?=uci-heart-failure
VERSION?=dev

help:
	@cat Makefile

setup-env:
	pip install -r requirements.txt

run-unit-tests: setup-env
	python -m unittest discover -s test -p 'test_*.py'

run-local: setup-env
	python main.py

build-docker:
	docker build \
		-t $(IMAGE_NAME):$(VERSION) \
		-f Dockerfile .

run-docker: build-docker
	docker run -it --rm \
		--name $(IMAGE_NAME) \
		-p 8000:8000 \
		$(IMAGE_NAME):$(VERSION)