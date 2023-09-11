# Fashion CLIP

## How To

<details><summary>setup API Gateway</summary>

```bash
sam build
sam local start-api -v ./app:/var/task
```

</details>

<details><summary>setup Lightsail</summary>

```bash
docker build -t fashion-clip:lightsail . -f ./Dockerfile.aws.lightsail

docker container run --rm \
    -v `pwd`/app:/app \
    -v `pwd`/layer:/opt \
    -e CLIP_MODEL_PATH=/opt/ml/clip \
    -p 8080:8000 \
    fashion-clip:lightsail --reload
```

</details>


<details><summary>setup SageMaker</summary>

```bash
docker build -t fashion-clip:sagemaker . -f ./Dockerfile.aws.sagemaker

docker container run --rm \
    -v `pwd`/app:/app \
    -p 8080:8080 \
    fashion-clip:sagemaker serve --local --port 8080
```

</details>

<details><summary>upload clip model to s3 bucket</summary>

```bash
cd layer
zip -r ml ml
aws s3 cp ml https://fashion-clip-model
```

</details>