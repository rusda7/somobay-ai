import os, re, json, pathlib, hashlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Somobay AI - Final 6 Rules Compliant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE = pathlib.Path(__file__).parent

def load_file(name):
    for p in [BASE/name, pathlib.Path(f"/mnt/data/{name}")]:
        if p.exists():
            try:
                # Try utf-8 first, then cp1252 fallback for broken fonts
                txt = p.read_text(encoding='utf-8')
                # Fix broken font: remove \r, normalize spaces
                txt = txt.replace('\r','').replace('\x00','')
                txt = re.sub(r'[ \t]+', ' ', txt)
                return txt
            except:
                try:
                    return p.read_text(encoding='utf-8', errors='ignore')
                except:
                    continue
    return ""

AIN = load_file("ain_2001.txt")
BIDI = load_file("bidhimala_2004.txt")
CIRC = load_file("circular.txt")

print(f"Loaded - AIN: {len(AIN)} BIDI: {len(BIDI)} CIRC: {len(CIRC)}")

# 03 - Font fix: Build clean index
def build_index(text):
    idx={}
    # Pattern for Bengali/English numbers
    matches=list(re.finditer(r'\n\s*([০-৯0-9]+)[।]\s*', text))
    for i,m in enumerate(matches):
        num_bn = m.group(1)
        num_en = num_bn.translate(str.maketrans("০১২৩৪৫৬৭৮৯","0123456789"))
        start=m.start()
        end=matches[i+1].start() if i+1 < len(matches) else start+12000
        chunk=text[start:end].strip()
        # Clean broken font characters
        chunk = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', chunk)
        if len(chunk)>100:
            idx[num_en]=chunk[:10000]
            idx[num_bn]=chunk[:10000]
    return idx

AIN_IDX = build_index(AIN)
BIDI_IDX = build_index(BIDI)

# Chunk for semantic search
def build_chunks(text, source):
    chunks=[]
    parts=re.split(r'\n\s*\n', text)
    buf=""
    for part in parts:
        part=part.strip()
        if len(part)<60: continue
        # Clean
        part=re.sub(r'\s+', ' ', part)
        if re.match(r'^[০-৯0-9]+[।]', part) and len(buf)>500:
            chunks.append({"text": buf, "source": source, "len": len(buf)})
            buf=part
        else:
            buf += "\n\n" + part
            if len(buf)>1400:
                chunks.append({"text": buf, "source": source, "len": len(buf)})
                buf=""
    if buf: chunks.append({"text": buf, "source": source, "len": len(buf)})
    return chunks

ALL_CHUNKS = build_chunks(AIN, "আইন") + build_chunks(BIDI, "বিধি")
if CIRC: ALL_CHUNKS+=build_chunks(CIRC, "সার্কুলার")

# 04 - Token save: Persistent cache on disk
CACHE_FILE = BASE/"cache.json"
CACHE={}
if CACHE_FILE.exists():
    try:
        CACHE=json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        print(f"Cache loaded {len(CACHE)} items")
    except: CACHE={}

def save_cache():
    try:
        CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False), encoding='utf-8')
    except: pass

def get_cache_key(q):
    return hashlib.md5(q.lower().strip().encode('utf-8')).hexdigest()

# Smart retriever
def retrieve(query):
    trans=str.maketrans("০১২৩৪৫৬৭৮৯","0123456789")
    # 1. Sub-dhara 50(6)
    m=re.search(r'([০-৯0-9]+)\s*\(\s*([০-৯0-9]+)\s*\)', query)
    if m:
        dhara_en=m.group(1).translate(trans)
        if dhara_en in AIN_IDX:
            return [AIN_IDX[dhara_en]], f"ধারা {m.group(1)}({m.group(2)})"
    # 2. Single number
    nums=[n.translate(trans) for n in re.findall(r'[০-৯0-9]+', query)]
    for n in nums:
        if n in AIN_IDX:
            return [AIN_IDX[n]], f"ধারা {n}"
        if n in BIDI_IDX:
            return [BIDI_IDX[n]], f"বিধি {n}"
    # 3. Keyword search - full book
    q_tokens=set(re.findall(r'[\u0980-\u09FF]{3,}', query))
    stop=set(["সমবায়","সমিতি","আইনে","ধারায়","বিধিতে","কি","কত","ব্যাখ্যা","বলুন","আছে","হবে","সম্পর্কে"])
    q_f=[t for t in q_tokens if t not in stop]
    if not q_f: q_f=list(q_tokens)
    scored=[]
    for ch in ALL_CHUNKS:
        txt=ch["text"]
        score=sum(1 for qt in q_f if qt in txt)
        # Boost if multiple keywords match
        if score>=1:
            scored.append((score, txt))
    scored.sort(reverse=True, key=lambda x:x[0])
    if not scored:
        return [], ""
    # Return top 3 chunks - full book reading
    top_texts=[t for _,t in scored[:3]]
    return top_texts, "প্রাসঙ্গিক অংশ"

GROQ_KEY=os.environ.get("GROQ_API_KEY","")
groq_client=None
if GROQ_KEY:
    try:
        from groq import Groq
        groq_client=Groq(api_key=GROQ_KEY)
    except Exception as e:
        print(f"Groq init fail {e}")

class ChatRequest(BaseModel):
    question: str
    user_id: str="anon"
class ChatResponse(BaseModel):
    answer: str
    references: List[str] = []
    cached: bool=False

@app.get("/")
def root():
    return {"status":"6 Rules Compliant", "rules": "01 ChatGPT-like, 02 Only txt files, 03 No font break, 04 Token save, 05 No cut, 06 Accurate", "cache": len(CACHE), "ain_chunks": len(AIN_IDX), "groq": bool(groq_client)}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q=req.question.strip()
    if not q:
        return ChatResponse(answer="প্রশ্ন লিখুন", references=[], cached=False)

    ckey=get_cache_key(q)
    if ckey in CACHE:
        return ChatResponse(answer=CACHE[ckey], references=[], cached=True)

    # 02 - Only from txt files
    chunks, ref_label = retrieve(q)
    
    if not chunks:
        ans="দুঃখিত, আপনার আপলোডকৃত ain_2001.txt, bidhimala_2004.txt এবং সার্কুলার ফাইলে এই বিষয়ে সরাসরি তথ্য খুঁজে পাওয়া যায়নি। অনুগ্রহ করে ধারা নম্বর দিয়ে জিজ্ঞাসা করুন।\n\nসূত্র: -"
        CACHE[ckey]=ans; save_cache()
        return ChatResponse(answer=ans, references=[], cached=False)

    context="\n\n---\n\n".join(chunks)[:15000]

    if groq_client:
        try:
            # 01,02,03,06 - ChatGPT like, only txt, no broken font, accurate
            system_prompt = """তুমি সমবায় আইন AI - ChatGPT এর মতো বন্ধুত্বপূর্ণ কিন্তু ১০০% নির্ভুল।

তোমার ৬টি কঠোর নিয়ম:
০১. ChatGPT এর মতো সুন্দর, গুছানো, মানবিক ভাষায় উত্তর দেবে।
০২. শুধুমাত্র ব্যবহারকারীর আপলোড করা ain_2001.txt, bidhimala_2004.txt এবং circular.txt থেকেই উত্তর দেবে। বাইরে থেকে কোনো তথ্য, ইতিহাস বা বানানো উত্তর দেবে না।
০৩. ফন্ট ভেঙে যাওয়া বা এলোমেলো লেখা দেবে না। পরিষ্কার বাংলা ইউনিকোডে উত্তর দেবে।
০৬. ভুল উত্তর একদম দেবে না। Context এ না থাকলে বলবে "এই বিষয়ে ফাইলে তথ্য নেই"।

উত্তরের ফরম্যাট (সব প্রশ্নের জন্য একই):
- প্রথম লাইন: সুনির্দিষ্ট সরাসরি উত্তর (যেমন: "এডহক কমিটির মেয়াদ ১২০ দিন।" বা "ব্যবস্থাপনা কমিটির মেয়াদ ৩ বছর।") শেষে (ধারা X) উল্লেখ।
- মাঝে ৩-৪ লাইনে বিস্তারিত ব্যাখ্যা: উপ-ধারা সহ (যেমন ১৮(৩), ১৮(৪), ১৮(৫), ১৮(৭)) বুঝিয়ে বলো।
- শেষ লাইন: "সূত্র: ধারা ১৮(৫), ১৮(৭)" এই ফরম্যাটে।

উত্তর ২০০ শব্দের মধ্যে রাখো যাতে কেটে না যায়, কিন্তু তথ্য অসম্পূর্ণ রেখো না।
"""

            user_prompt = f"""Context (আপনার আপলোড করা ৩টি ফাইল থেকে হুবহু অংশ):
{context}

প্রশ্ন: {q}

উপরের Context থেকেই শুধুমাত্র নির্ভুল উত্তর দাও। Context এ না থাকলে "ফাইলে তথ্য নেই" বলো।"""

            comp=groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_prompt}
                ],
                temperature=0.05,
                max_tokens=700,
            )
            final=comp.choices[0].message.content.strip()
            # 03 - Final font clean
            final=re.sub(r'[ \t]+', ' ', final)
            final=final.replace('�','')

            CACHE[ckey]=final; save_cache()
            return ChatResponse(answer=final, references=[], cached=False)
        except Exception as e:
            print(f"Groq error {e}")
            import traceback; traceback.print_exc()

    # Fallback without Groq - direct from txt
    ans = chunks[0][:1500] + f"\n\nসূত্র: {ref_label}"
    CACHE[ckey]=ans; save_cache()
    return ChatResponse(answer=ans, references=[], cached=False)

@app.get("/api/clear-cache")
def clear_cache():
    global CACHE
    CACHE={}
    save_cache()
    return {"cleared": True}
