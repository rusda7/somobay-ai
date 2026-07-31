from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, json, hashlib, re
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="CoopBd AI - Final Gemini+PDF", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Try Gemini first, fallback to Groq ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

gemini_model = None
groq_client = None

if GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print(f"✅ Gemini 1.5 Flash Ready")
    except Exception as e:
        print(f"Gemini Error: {e}")

if not gemini_model and GROQ_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_KEY)
        print(f"✅ Groq Ready as fallback")
    except Exception as e:
        print(f"Groq Error: {e}")

# --- Load PDF or TXT (PDF first for correct font) ---
law_book = ""
pdf_chunks = []

def load_files():
    global law_book, pdf_chunks
    # Try PDF first (your old correct method)
    try:
        from pypdf import PdfReader
        pdf_files = ["ain_2001.pdf", "bidhimala_2004.pdf", "circular.pdf"]
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                try:
                    reader = PdfReader(pdf_file)
                    full_text = ""
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        full_text += page_text + "\n"
                        # Your old chunking - 800 chars
                        for i in range(0, len(page_text), 800):
                            chunk = page_text[i:i+800]
                            if len(chunk.strip()) > 100:
                                pdf_chunks.append({
                                    "text": chunk,
                                    "source": pdf_file,
                                    "page": page_num + 1
                                })
                    law_book += f"\n\n--- {pdf_file} ---\n\n" + full_text
                    print(f"✅ {pdf_file} PDF Loaded, Pages: {len(reader.pages)}")
                except Exception as e:
                    print(f"PDF Error {pdf_file}: {e}")
    except ImportError:
        print("pypdf not installed, trying txt")

    # Fallback to TXT if no PDF chunks
    if not pdf_chunks:
        for txt_file in ["ain_2001.txt", "bidhimala_2004.txt", "circular.txt"]:
            if os.path.exists(txt_file):
                try:
                    text = open(txt_file, encoding='utf-8', errors='ignore').read()
                    law_book += f"\n\n--- {txt_file} ---\n\n" + text
                    for i in range(0, len(text), 800):
                        chunk = text[i:i+800]
                        if len(chunk.strip()) > 100:
                            pdf_chunks.append({
                                "text": chunk,
                                "source": txt_file,
                                "page": i//800 + 1
                            })
                    print(f"✅ {txt_file} TXT Loaded")
                except Exception as e:
                    print(f"TXT Error {txt_file}: {e}")

    print(f"Total: {len(law_book)} chars, {len(pdf_chunks)} chunks")

load_files()

# --- Token Save Cache (Your Rule 04) ---
CACHE_FILE = "cache.json"
CACHE = {}
if os.path.exists(CACHE_FILE):
    try:
        CACHE = json.loads(open(CACHE_FILE, encoding='utf-8').read())
        print(f"Cache loaded: {len(CACHE)}")
    except: CACHE={}

def save_cache():
    try:
        open(CACHE_FILE, 'w', encoding='utf-8').write(json.dumps(CACHE, ensure_ascii=False))
    except: pass

def ckey(q): return hashlib.md5(q.lower().strip().encode('utf-8')).hexdigest()

# --- Synonym Map for Accurate Retrieval (Your corrections) ---
SYNONYM_BOOST = {
    "অবসায়ন": ["অবসায়ন", "অবসায়ক", "৫৩", "৫৪", "৫৫", "৫৬", "৫৭", "৫৮"],
    "অবসায়ন": ["অবসায়ন", "৫৩", "৫৪", "৫৫", "৫৬", "৫৭", "৫৮"],
    "নির্বাচন না করলে": ["নির্বাচন", "বিলুপ্ত", "১৮", "১২০", "অন্তর্বর্তী"],
    "শাস্তি": ["বিলুপ্ত", "ভঙ্গ", "১৮", "২২"],
    "এডহক": ["অন্তর্বর্তী", "১৮", "১২০"],
    "মেয়াদ শেষ": ["মেয়াদ", "১৮", "বিলুপ্ত"],
}

class QuestionRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    question: str
    user_id: str = "anon"

class ChatResponse(BaseModel):
    answer: str
    references: List[str] = []
    cached: bool = False
    sources: List[str] = []

from typing import List

def retrieve(query, top_k=5):
    # Expand query with synonyms
    expanded_query = query
    for key, synonyms in SYNONYM_BOOST.items():
        if key in query:
            expanded_query += " " + " ".join(synonyms)
    
    query_words = [w for w in expanded_query.split() if len(w) > 2]
    
    scored = []
    for chunk in pdf_chunks:
        score = 0
        chunk_lower = chunk["text"].lower()
        for word in query_words:
            if word.lower() in chunk_lower:
                score += 1
        # Bonus for exact dhara number match like ৫৩, ৫৪
        if score > 0:
            scored.append({"score": score, "data": chunk})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "CoopBd AI Backend চালু আছে - Final", "chunks": len(pdf_chunks), "gemini": bool(gemini_model), "groq": bool(groq_client), "cache": len(CACHE)}

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(pdf_chunks), "gemini": bool(gemini_model), "cache": len(CACHE)}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    q = req.text.strip()
    k = ckey(q)
    if k in CACHE:
        return {"answer": CACHE[k], "sources": [], "cached": True}

    top_chunks = retrieve(q, top_k=5)
    
    context = ""
    sources = []
    for item in top_chunks:
        context += f"\n\n{item['data']['text']}"
        sources.append(f"{item['data']['source']}, পৃষ্ঠা {item['data']['page']}")

    if not context:
        context = "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"

    # === YOUR OLD PROMPT - Upgraded with your 6 rules ===
    prompt = f"""আপনি CoopBd AI, বাংলাদেশ সমবায় আইন ২০০১ ও বিধিমালা ২০০৪ বিশেষজ্ঞ।

আপনার ৬টি কঠোর নিয়ম:
০১. ChatGPT এর মতো সুন্দর, গুছানো, মানবিক ভাষায় উত্তর দেবেন।
০২. শুধুমাত্র নিচের আইনের অংশগুলো (ain_2001.pdf, bidhimala_2004.pdf) থেকেই উত্তর দেবেন। বাইরে থেকে বানানো উত্তর দেবেন না।
০৩. ফন্ট ভাঙবে না, পরিষ্কার বাংলায় উত্তর দেবেন।
০৪. রিপিট প্রশ্নে টোকেন বাঁচান।
০৫. উত্তর বড় হলেও সম্পূর্ণ দেখাবেন, কেটে দেবেন না।
০৬. নির্ভুল উত্তর খুবই জরুরী। ভুল উত্তর দেবেন না।

বিশেষ তথ্য (ব্যবহারকারীর কারেকশন):
- অবসায়ন মূলত ধারা ৫৩-৫৮ এবং বিধিমালা ৮০-৮৫
- এডহক/অন্তর্বর্তী কমিটির মেয়াদ ১২০ দিন (ধারা ১৮(৫))
- ১৮(৩) প্রথম কমিটি ২ বছর, ১৮(৪) নির্বাচিত কমিটি ৩ বছর, ১৮(৫) না করলে বিলুপ্ত ও ১২০ দিনের অন্তর্বর্তী, ১৮(৭) পুনরায় অন্তর্বর্তী

আইনের অংশ:
{context[:12000]}

প্রশ্ন: {q}

উত্তরের ফরম্যাট:
১. প্রথম লাইনে সুনির্দিষ্ট সরাসরি উত্তর (যেমন: "অবসায়ন ধারা ৫৩-৫৮ এ বর্ণিত" বা "মেয়াদ ১২০ দিন")
২. মাঝে ৩-৪ বাক্যে বিস্তারিত ব্যাখ্যা
৩. শেষে "সূত্র: ধারা ৫৩-৫৮, বিধি ৮০-৮৫" এইভাবে

উত্তর:"""

    try:
        answer_text = ""
        if gemini_model:
            response = gemini_model.generate_content(prompt)
            answer_text = response.text
        elif groq_client:
            comp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content": prompt}],
                temperature=0.05,
                max_tokens=800,
            )
            answer_text = comp.choices[0].message.content
        else:
            answer_text = "API Key পাওয়া যায়নি।"

        CACHE[k] = answer_text
        save_cache()

        return {"answer": answer_text, "sources": list(set(sources)), "cached": False}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": f"Error: {str(e)}", "sources": []}

# --- New endpoint for your new frontend (keeps old /ask working) ---
@app.post("/api/chat")
def chat_api(req: ChatRequest):
    # Reuse /ask logic
    q_req = QuestionRequest(text=req.question)
    result = ask_question(q_req)
    # Convert to new format (no yellow box)
    return ChatResponse(
        answer=result["answer"],
        references=[],  # No yellow box as you wanted
        cached=result.get("cached", False),
        sources=result.get("sources", [])
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
