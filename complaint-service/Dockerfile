FROM python:3.10-slim

# إعداد مستخدم غير جذر (Non-root) لتوافق حماية Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# نسخ ملف المتطلبات أولاً
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# التثبيت الذكي: السيرفر الرئيسي للحزم العادية، والسيرفر الإضافي للـ Torch CPU
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# نسخ باقي كود السيرفس
COPY --chown=user . $HOME/app

# كشف البورت الإلزامي لـ Hugging Face
EXPOSE 7860

CMD ["python", "main.py"]