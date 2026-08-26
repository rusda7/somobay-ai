
import os, re, pickle, json
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
EMB_FILE = "embeddings.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. বাংলা আইনের জন্য স্মার্ট চাঙ্কিং ---
def chunk_by_dhara(text):
    # ধারা, বিধি, উপ-বিধি দিয়ে ভাগ করা
    pattern = r'(?=(?:ধারা|বিধি|উপ-বিধি|অনুচ্ছেদ)\s*[০-৯0-9]+)'
    parts = re.split(pattern, text)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) < 50: 
            continue
        # বড় ধারা হলে 600 শব্দে ভাগ + 100 শব্দ overlap
        words = p.split()
        if len(words) > 600:
            for i in range(0, len(words), 500):
                chunk = " ".join(words[i:i+600])
                if len(chunk) > 100:
                    chunks.append(chunk)
        else:
            chunks.append(p)
    return chunks

def get_text_from_file(path):
    if path.endswith(".pdf"):
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def create_embeddings():
    all_chunks = []
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.isfile(fpath): continue
        try:
            raw = get_text_from_file(fpath)
            chunks = chunk_by_dhara(raw)
            for ch in chunks:
                all_chunks.append({"text": ch, "source": fname})
        except Exception as e:
            print(f"Error {fname}: {e}")

    print(f"Total chunks: {len(all_chunks)}")
    # Embedding
    texts = [c["text"] for c in all_chunks]
    embeddings = []
    # Gemini embedding batch
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = genai.embed_content(model="models/text-embedding-004", content=batch, task_type="retrieval_document")
        embeddings.extend(resp["embedding"])
    
    data = {"chunks": all_chunks, "embeddings": np.array(embeddings)}
    with open(EMB_FILE, "wb") as f:
        pickle.dump(data, f)
    return len(all_chunks)

def search(query, top_k=5):
    if not os.path.exists(EMB_FILE):
        return []
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)
    
    q_emb = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")["embedding"]
    q_emb = np.array(q_emb)
    
    # Cosine similarity
    emb_matrix = data["embeddings"]
    sims = np.dot(emb_matrix, q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        if score < 0.35: # Zero-hallucination lock
            continue
        results.append({
            "text": data["chunks"][idx]["text"],
            "source": data["chunks"][idx]["source"],
            "score": score
        })
    return results

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin", methods=["GET","POST"])
def admin():
    if request.method == "POST":
        files = request.files.getlist("files")
        for file in files:
            if file.filename:
                file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        count = create_embeddings()
        return f"<h3>{len(files)} টি ফাইল আপলোড হয়েছে। {count} টি ধারা তৈরি হয়েছে। <a href='/admin'>ফিরে যান</a></h3>"
    file_list = os.listdir(UPLOAD_FOLDER)
    has_emb = os.path.exists(EMB_FILE)
    return render_template("admin.html", files=file_list, has_emb=has_emb)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question","").strip()
    if not question:
        return jsonify({"answer":"প্রশ্ন লিখুন"})
    
    contexts = search(question, top_k=5)
    if not contexts:
        return jsonify({"answer":"দুঃখিত, এই তথ্যটি আমার কাছে থাকা সমবায় আইনের ডকুমেন্টে নেই।", "sources":[]})
    
    context_text = "\n\n".join([f"[Source: {c['source']} Score: {c['score']:.2f}]\n{c['text']}" for c in contexts])
    
    prompt = f"""
তুমি বাংলাদেশের সমবায় আইন ও বিধিমালা বিশেষজ্ঞ।
নিচে CONTEXT দেওয়া হলো যা ব্যবহারকারীর আপলোড করা ডকুমেন্ট থেকে এসেছে।

নিয়ম:
1. শুধুমাত্র CONTEXT থেকে উত্তর দেবে। নিজের সাধারণ জ্ঞান ব্যবহার করবে না।
2. CONTEXT এ উত্তর না থাকলে সরাসরি বলবে: "এই তথ্যটি প্রদত্ত ডকুমেন্টে নেই।"
3. উত্তর বাংলায়, সহজ ভাষায় দেবে। প্রয়োজনে ধারা নম্বর উল্লেখ করবে।
4. আইনের মতো জটিল বিষয় বানিয়ে বলবে না।

CONTEXT:
{context_text}

প্রশ্ন: {question}

উত্তর:
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    return jsonify({
        "answer": response.text,
        "sources": contexts
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
