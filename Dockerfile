# Candidate base; container certification is blocked on the development host.
FROM vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b AS wheel
WORKDIR /build
COPY compat/build-requirements.txt /build/build-requirements.txt
RUN python3 -c "import sys; assert sys.version_info[:2] == (3, 12)" && \
    python3 -m pip install --require-hashes -r build-requirements.txt
COPY pyproject.toml README.md LICENSE NOTICE /build/
COPY licenses /build/licenses
COPY src /build/src
RUN python3 -m hatchling build -t wheel

FROM vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
WORKDIR /app
COPY compat/runtime-requirements.txt /app/runtime-requirements.txt
RUN python3 -c "import sys; assert sys.version_info[:2] == (3, 12)" && \
    python3 -m pip install --require-hashes -r runtime-requirements.txt && \
    mkdir -p /cache && chown 10001:10001 /cache
COPY --from=wheel /build/dist/*.whl /app/
RUN python3 -m pip install --no-deps /app/*.whl
ENV JEVSERVE_CACHE_DIR=/cache JEVSERVE_HOST=0.0.0.0 \
    VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20m --retries=4 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
ENTRYPOINT ["jevserve"]
CMD ["serve"]
