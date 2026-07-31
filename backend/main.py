import os, re, json, pathlib, hashlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Somobay AI - Accurate Synonym Fixed")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE = pathlib.Path(__file__).parent
def load_file(name):
    for p in [BASE/name, pathlib.Path(f"/mnt/data/{name}")]:
        if p.exists():
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore')
                txt = re.sub(r'\r', '', txt)
                return txt
            except: continue
    return ""

AIN = load_file("ain_2001.txt")
BIDI = load_file("bidhimala_2004.txt")
CIRC = load_file("circular.txt")

def build_index(text):
    idx={}
    matches=list(re.finditer(r'\n\s*([০-৯0-9]+)[।]\s*', text))
    for i,m in enumerate(matches):
        num_en = m.group(1).translate(str.maketrans("০১২৩৪৫৬৭৮৯","0123456789"))
        start=m.start()
        end=matches[i+1].start() if i+1 < len(matches) else start+12000
        chunk=text[start:end].strip()[:10000]
        if len(chunk)>150:
            idx[num_en]=chunk
    return idx

AIN_IDX = build_index(AIN)
BIDI_IDX = build_index(BIDI)

print(f"Index - AIN {len(AIN_IDX)} BIDI {len(BIDI_IDX)}")

# === COMPREHENSIVE SYNONYM MAP - This fixes "প্রশ্ন করি কি উত্তর দেয় কি" ===
SYNONYM_MAP = {
    # নির্বাচন সংক্রান্ত - সবগুলো ১৮ ধারায় যাবে
    "নির্বাচন না করলে": "18",
    "নির্বাচন না হলে": "18",
    "নির্বাচন না": "18",
    "নির্বাচন করতে না পারলে": "18",
    "শাস্তি": "18,22",
    "বিলুপ্ত": "18",
    "ভঙ্গ": "22",
    "মেয়াদ শেষ": "18",
    "মেয়াদ শেষ": "18",
    "মেয়াদ শেষ হলে": "18",
    "করণীয় কি": "18,20",
    "করণীয়": "18,20",
    "এডহক": "18",
    "অন্তর্বর্তী": "18",
    "অন্তবর্তী": "18",
    "মেয়াদ কত": "18",
    "মেয়াদ কত": "18",
    "কতদিন": "18,19,20",
    "পদ শূন্য": "20",
    "শূন্য হলে": "20",
    "যোগ্যতা": "19",
    "অযোগ্যতা": "19",
    "বিরোধ": "50",
    "বিবাদ": "50",
    "সভা না করলে": "22,23",
    "সাধারণ সভা": "23",
    "বিশেষ সভা": "24",
    "নিবন্ধন": "9,10,11",
    "বাতিল": "17,22",
    "সদস্য পদ বাতিল": "17",
    "ঋণ": "87,88",
    "ঋণ খেলাপি": "19,87",
    "অডিট": "40,41,42,43,44",
    "নিরীক্ষা": "40,41",
    "ডিগ্রি": "42",
    "সার্টিফিকেট": "52,53",
    "অবসায়ন": "62,63",
    "তদন্ত": "49",
    "পরিদর্শন": "48",
    "উপ-আইন": "15,16",
}

def make_chunks(text):
    chunks=[]
    for para in re.split(r'\n\s*\n', text):
        para=para.strip()
        if len(para)>80:
            chunks.append(re.sub(r'\s+', ' ', para))
    return chunks

ALL_CHUNKS = make_chunks(AIN) + make_chunks(BIDI)
if CIRC: ALL_CHUNKS+=make_chunks(CIRC)

def retrieve_accurate(query):
    trans=str.maketrans("০১২৩৪৫৬৭৮৯","0123456789")
    q_lower=query.lower()
    
    # 1. Sub-dhara 50(6)
    m=re.search(r'([০-৯0-9]+)\s*\(\s*([০-৯0-9]+)\s*\)', query)
    if m:
        dhara_en=m.group(1).translate(trans)
        if dhara_en in AIN_IDX:
            return [AIN_IDX[dhara_en]], f"ধারা {m.group(1)}({m.group(2)})"
    
    # 2. Synonym map - check longest phrase first
    sorted_synonyms=sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
    for phrase in sorted_synonyms:
        if phrase in query:
            dhara_list = SYNONYM_MAP[phrase].split(',')
            result=[]
            for d in dhara_list:
                d=d.strip()
                if d in AIN_IDX:
                    result.append(AIN_IDX[d])
            if result:
                return result[:2], f"ধারা {SYNONYM_MAP[phrase]}"

    # 3. Direct number
    nums=[n.translate(trans) for n in re.findall(r'[০-৯0-9]+', query)]
    for n in nums:
        if n in AIN_IDX:
            return [AIN_IDX[n]], f"ধারা {n}"
    
    # 4. Keyword fallback
    q_tokens=set(re.findall(r'[\u0980-\u09FF]{3,}', query))
    stop=set(["সমবায়","সমিতি","আইনে","ধারায়","কি","কত","হবে","সম্পর্কে","বলুন","জানতে","চাই"])
    q_f=[t for t in q_tokens if t not in stop]
    scored=[]
    for ch in ALL_CHUNKS:
        score=sum(1 for qt in q_f if qt in ch)
        if score>0:
            scored.append((score, ch))
    scored.sort(reverse=True, key=lambda x:x[0])
    if scored:
        return [t for _,t in scored[:2]], "প্রাসঙ্গিক অংশ"
    return [], ""

GROQ_KEY=os.environ.get("GROQ_API_KEY","")
groq_client=None
if GROQ_KEY:
    try:
        from groq import Groq
        groq_client=Groq(api_key=GROQ_KEY)
    except: pass

CACHE_FILE=BASE/"cache.json"
CACHE={}
if CACHE_FILE.exists():
    try: CACHE=json.loads(CACHE_FILE.read_text(encoding='utf-8'))
    except: CACHE={}

def save_cache():
    try: CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False), encoding='utf-8')
    except: pass

def ckey(q): return hashlib.md5(q.lower().strip().encode('utf-8')).hexdigest()

class ChatRequest(BaseModel):
    question: str
    user_id: str="anon"
class ChatResponse(BaseModel):
    answer: str
    references: List[str] = []
    cached: bool=False

@app.get("/")
def root():
    return {"status":"Synonym Accurate", "cache": len(CACHE), "groq": bool(groq_client)}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q=req.question.strip()
    k=ckey(q)
    if k in CACHE:
        return ChatResponse(answer=CACHE[k], references=[], cached=True)

    if "অধ্যায়" in q and "ধারা" in q:
        ans="সমবায় সমিতি আইন, ২০০১ এ ১৩টি অধ্যায় ও ৯০টি ধারা রয়েছে। বিধিমালা ২০০৪ এ ১১৭টি বিধি রয়েছে।\n\nসূত্র: আইন ২০০১ ভূমিকা"
        CACHE[k]=ans; save_cache()
        return ChatResponse(answer=ans, references=[], cached=False)

    chunks, ref = retrieve_accurate(q)
    
    if not chunks:
        ans=f"দুঃখিত, '{q}' বিষয়ে আপনার আপলোড করা ফাইলে সরাসরি তথ্য পাওয়া যায়নি। ধারা নম্বর দিয়ে জিজ্ঞাসা করুন।\n\nসূত্র: -"
        CACHE[k]=ans; save_cache()
        return ChatResponse(answer=ans, references=[], cached=False)

    context="\n\n---\n\n".join(chunks)[:15000]

    if groq_client:
        try:
            system="""তুমি সমবায় আইন AI। তোমার কাজ ১০০% নির্ভুল উত্তর দেওয়া।

কঠোর নিয়ম:
১. শুধুমাত্র Context (ain_2001.txt, bidhimala_2004.txt, circular.txt) থেকেই উত্তর দেবে। বাইরে থেকে বানাবে না।
২. প্রশ্নের সুনির্দিষ্ট উত্তর প্রথম লাইনে দাও। যেমন:
   - "নির্বাচন না করলে কমিটি বিলুপ্ত হবে এবং ১২০ দিনের জন্য অন্তর্বর্তী কমিটি নিয়োগ হবে।"
   - "এডহক কমিটির মেয়াদ ১২০ দিন।"
৩. তারপর ৩-৪ বাক্যে আইনের ভাষায় ব্যাখ্যা দাও, উপ-ধারা সহ (১৮(৩)(৪)(৫)(৭) ইত্যাদি)।
৪. ভুল তথ্য, বাড়ি ভাড়া, বা অপ্রাসঙ্গিক তথ্য দেবে না।
৫. শেষ লাইনে "সূত্র: ধারা ১৮(৫)" লিখবে।
৬. উত্তর বাংলায়, পরিষ্কার, ২০০ শব্দের মধ্যে।

যদি Context এ উত্তর না থাকে, বলো "এই বিষয়ে ফাইলে তথ্য নেই"।
"""

            user_p=f"""Context (আইনের হুবহু অংশ, শুধু এখান থেকেই উত্তর দাও):
{context}

প্রশ্ন: {q}

নির্ভুল উত্তর দাও:"""

            comp=groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user_p}
                ],
                temperature=0.02,
                max_tokens=700,
            )
            final=comp.choices[0].message.content.strip()
            CACHE[k]=final; save_cache()
            return ChatResponse(answer=final, references=[], cached=False)
        except Exception as e:
            print(e)

    ans=chunks[0][:1500] + f"\n\nসূত্র: {ref}"
    CACHE[k]=ans; save_cache()
    return ChatResponse(answer=ans, references=[], cached=False)
