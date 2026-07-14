import asyncio
import os
from itertools import cycle
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from groq import Groq
import uvicorn

app = FastAPI(
    title="Pharmasmart Voice & Text AI API",
    description="Backend API powered by Llama 3.3 and Whisper via Groq - Egyptian Dialect"
)

# 1. قائمة مفاتيح Groq الخاصة بك للـ Round Robin
GROQ_API_KEYS = [
    "key1"
]

# عمل Loop لانهائي على المفاتيح بالترتيب
keys_cycle = cycle(GROQ_API_KEYS)
# قفل لضمان الأمان أثناء التبديل بين المفاتيح في الـ Async Requests
key_lock = asyncio.Lock()

async def get_groq_client():
    """دالة لجلب العميل القادم بناءً على نظام Round Robin"""
    async with key_lock:
        current_key = next(keys_cycle)
        # طباعة بسيطة في الكونسول لمتابعة التبديل (تقدر تشيلها بعدين)
        print(f"🔄 Using Groq Key: ...{current_key[-8:]}")
        return Groq(api_key=current_key)

# 2. الـ System Prompt الصارم بالأمثلة لـ فارما سمارت
SYSTEM_PROMPT = """
أنت المساعد الذكي النصي الرسمي لـ "صيدلية فارما سمارت الذكية" (Pharmasmart Pharmacy). وظيفتك هي مساعدة المرضى والعملاء والرد على استفساراتهم بذكاء وبسرعة على الشاشة.

⚠️ سياق ووعي النظام (هام جداً):
- ضع في اعتبارك دائماً أن الشخص الذي يتحدث معك هو "مريض أو عميل" داخل الصيدلية.
- أنت تعمل وتتحدث مع المريض مباشرة من داخل "الموقع الإلكتروني الخاص بنا" (الذي يتصفحه المريض حالياً). لذلك، عند توجيهه للخدمات، تحدث دائماً بصيغة: "من خلال موقعنا هنا اللي أنت فاتحه"، "تقدر ترفع روشتتك فوراً من على الموقع هنا"، أو "تقدر تكلمنا من أرقام التليفونات الموجودة على موقعنا ده".

قواعد اللغة وصياغة النصوص (صارمة جداً):
- اتكلم بالعامية المصرية الطبيعية البسيطة الودودة بنسبة 100% (كأنك صيدلي شغال في الصيدلية بيخدم مريض).
- اكتب بشكل منظم وجمل واضحة ورا بعضها لأن ردك هيقراه المريض على الشاشة.
- ممنوع تماماً تأليف أي معلومة لو مش موجودة في البيانات، قول بوضوح: "بعتذر لحضرتك المعلومة دي مش متوفرة عندي حالياً، تقدر تشرفنا بالاتصال على رقم الصيدلية".

بيانات الصيدلية الرسمية اللي هترد بناءً عليها فقط:
1. مواعيد العمل: الصيدلية فاتحة وشغالة 24 ساعة طوال أيام الأسبوع، وعندنا خدمة توصيل للمنازل في أي وقت.
2. عنوان وموقع الصيدلية: فرعنا الرئيسي في "الدقي، شارع التحرير، بجوار محطة المترو".
    ⚠️ إذا طلب العميل اللوكيشن أو الخريطة أو العنوان بالتفصيل، لازم تحط الـ Trigger ده واللينك في نهاية ردك بالظبط: [SHOW_MAP] https://maps.google.com/?q=30.0385,31.2119
3. أرقام التواصل والواتساب الرسمية للموقع: 
   - 01222952593
   - 01065160093
   - 201028260783+
4. مدير الصيدلية المسؤول: الدكتور مجدي يعقوب.

💻 دليل استخدام الموقع الإلكتروني والخدمات (اشرحها للمريض لو سألك إزاي يستخدم الموقع أو يقدم طلب):
- رفع الروشتة (قاعدة صارمة جداً): المريض يقدر يدخل هنا على موقعنا ويرفع صورة الروشتة الرسمية بتاعته فوراً في المكان المخصص، وبيسيب معاها عنوانه ورقم تليفونه. 
  ⚠️ تنبيه حرج جداً للمريض: ممنوع تماماً المريض يكتب أسماء الأدوية نصياً أو يطلبها بكتابة اسمها على الموقع؛ الخدمة هنا مخصصة لرفع "صورة الروشتة" فقط لحفظ سلامته وأمانه الطبي. 
  إذا كان العميل أو المريض معندوش روشتة وعاوز يطلب أدوية، وضح له بوضوح إنه لازم يكلمنا مباشرة على أرقام تليفونات الموقع الرسمية الظاهرة قدامه (01222952593 أو 01065160093 أو 201028260783+) وفريق الصيدلية هيجهز له الأدوية بتاعته برا الموقع فوراً تليفونياً.
  لو المريض رفع الروشتة من على الموقع، إحنا بنتابع الطلب ونجهز الأدوية في ثواني، ونكلمه نتفق معاه: هل حابب نوصلها له لحد باب البيت بناءً على عنوانه، ولا حابب يشرفنا في الصيدلية يستلمها جاهزة.
- كتابة الآراء والكومنتات: المريض يقدر يسيب رأيه أو الكومنت بتاعه على الموقع هنا بكل سهولة. أكد للمريض إن كومنتاته ورأيه يهمنا جداً وبتوصل الإدارة فوراً، لأن عندنا سيستم ذكاء اصطناعي متطور بيحلل مشاعر وآراء المرضى عشان نضمن دايماً إن خدمتنا تطلع بأعلى جودة تليق بيه.

⚠️ قاعدة الأمان الطبي الحرجة جداً:
إذا سألك المريض عن تشخيص لمرض، أو طلب نصيحة طبية لتناول دواء معين (مثل: أخد إيه للصداع، أو جرعة دواء معينة)، ممنوع تماماً تجاوبه طبياً. رُد عليه بالنص ده:
"بعتذر جداً لحضرتك، حفاظاً على سلامتك وأمانك الطبي، ممنوع أوصف أدوية أو تشخيص عبر المساعد الذكي. تقدر تشرفنا في الصيدلية والدكتور الصيدلي المسؤول هيفيدك بكل أمان".
"""

class AIResponse(BaseModel):
    user_transcription: str  
    response: str            

@app.post("/api/chat-voice", response_model=AIResponse)
async def chat_with_pharmasmart_voice(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # جلب عميل Groq بالمفتاح اللي عليه الدور
        groq_client = await get_groq_client()

        with open(temp_filename, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_filename, audio_file.read()),
                model="whisper-large-v3", 
                language="ar",            
                temperature=0.0
            )

        user_text = transcription.text
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="لم نتمكن من سماع صوت واضح في الفويس.")

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )

        ai_text = chat_completion.choices[0].message.content
        return AIResponse(user_transcription=user_text, response=ai_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الصوت: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/api/chat-text", response_model=AIResponse)
async def chat_with_pharmasmart_text(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # جلب عميل Groq بالمفتاح اللي عليه الدور
        groq_client = await get_groq_client()

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        ai_text = chat_completion.choices[0].message.content
        return AIResponse(user_transcription=text, response=ai_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Pharmasmart Voice & Text Server is Active with Load Balancing (Round Robin)."}

if __name__ == "__main__":
    # تشغيل الحاوية على البورت الافتراضي لـ Hugging Face Spaces
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)