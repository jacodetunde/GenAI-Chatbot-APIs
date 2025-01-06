FROM public.ecr.aws/docker/library/python:3.9-slim-bullseye
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter


EXPOSE 8080

ARG OPENAI_API_KEY

# Set environment variables
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV AWS_LWA_INVOKE_MODE=RESPONSE_STREAM


COPY . .
WORKDIR .
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
