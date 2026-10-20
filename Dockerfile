FROM python:3.12-slim
# GPU: docker build --build-arg PADDLE=gpu .   (needs the NVIDIA container toolkit at run time)
ARG PADDLE=cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir "numpy>=2.1" \
    && if [ "$PADDLE" = gpu ]; then \
         pip install --no-cache-dir paddlepaddle-gpu==3.2.2 \
           --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/; \
       else pip install --no-cache-dir paddlepaddle==3.2.2; fi \
    && pip install --no-cache-dir . pytest httpx ruff
COPY src ./src
COPY tests ./tests
# Bake the models into the image so the service runs offline on-prem.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv5_server_det', text_recognition_model_name='th_PP-OCRv5_mobile_rec', use_doc_orientation_classify=True, use_textline_orientation=True, device='cpu', enable_mkldnn=False)"
ENV OCR_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
