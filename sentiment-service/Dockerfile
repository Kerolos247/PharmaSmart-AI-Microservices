FROM python:3.10-slim


RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app


RUN pip install --no-cache-dir --user torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu


COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt


COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "main.py"]
