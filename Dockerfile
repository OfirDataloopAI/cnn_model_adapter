FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:latest

USER 1000
ENV HOME=/tmp

RUN pip3 install https://storage.googleapis.com/dtlpy/dev/dtlpy-1.57.13-py3-none-any.whl --upgrade --user
RUN pip3 install https://storage.googleapis.com/dtlpy/agent/dtlpy_agent-1.57.13.0-py3-none-any.whl --upgrade --user
RUN pip3 install --user \
    torch \
    torchvision \
    imgaug \
    'scikit-image==0.17.2'


# docker build -t gcr.io/viewo-g/modelmgmt/resnet:0.0.6 -f ./Dockerfile  .
# docker push gcr.io/viewo-g/modelmgmt/resnet:0.0.6