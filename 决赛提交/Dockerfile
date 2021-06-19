FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# prepare your environment here
ENV LANG "en_US.UTF-8"
COPY . /app
WORKDIR /app/sohu_matching/src

# RUN pip install ...
RUN pip install transformers && pip install pandas && pip install scikit-learn

ENTRYPOINT ["python","infer_final.py"]