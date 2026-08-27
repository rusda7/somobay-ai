
import os, re, pickle, traceback
import numpy as np
from flask import Flask, request, render_template, jsonify
from google import genai
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
EMB_FILE = "embeddings.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_embedding(text):
    models_to_try = ["gemini-embedding-001", "text-embedding-004", "models/text-embedding-004", "models/gemini-embedding-001", "text-embedding-005"]
    last_err = None
    for m in models_to_try:
        try:
            resp = client.models.embed_content(model=m, contents=text)
            return resp.embeddings[0].values
        except Exception as e:
            last_err = e
            print(f"Embedding failed {m}: {e}")
            continue
    raise last_err

def chunk_by_dhara(text):
    import re
    pattern = r'(?:ধারা|বিধি|উপ-বিধি)\s*[০-৯0-9]+'
    parts = re.split(pattern, text)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) < 50: continue
        chunks.append(p)
    if not chunks and len(text) > 100:
        words = text.split()
        for i in range(0, len(words), 400):
            chunks.append(" ".join(words[i:i+500]))
    return chunks[:200]

def get_text_from_file(path):
    if path.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages[:30]:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def create_embeddings():
    all_chunks = []
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.isfile(fpath): continue
        if fname.startswith("."): continue
        raw = get_text_from_file(fpath)
        if not raw or len(raw) < 20: continue
        chunks = chunk_by_dhara(raw)
        for ch in chunks:
            all_chunks.append({"text": ch, "source": fname})
    if not all_chunks:
        raise ValueError("কোনো টেক্সট পাওয়া যায়নি।")
    embeddings = []
    for i, c in enumerate(all_chunks):
        emb = get_embedding(c["text"])
        embeddings.append(emb)
    data = {"chunks": all_chunks, "embeddings": np.array(embeddings)}
    with open(EMB_FILE, "wb") as f:
        pickle.dump(data, f)
    return len(all_chunks)

def search(query, top_k=5):
    if not os.path.exists(EMB_FILE): return []
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)
    q_emb = np.array(get_embedding(query))
    emb_matrix = data["embeddings"]
    sims = np.dot(emb_matrix, q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if float(sims[idx]) < 0.30: continue
        results.append({"text": data["chunks"][idx]["text"], "source": data["chunks"][idx]["source"], "score": float(sims[idx])})
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug")
def debug():
    try:
        models = client.models.list()
        model_list = []
        for m in models:
            model_list.append({"name": m.name, "methods": getattr(m, 'supported_actions', [])})
        return jsonify({"api_key_exists": bool(api_key), "models": model_list[:40]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "api_key_exists": bool(api_key)})

@app.route("/admin", methods=["GET","POST"])
def admin():
    error_msg = None
    success_msg = None
    if request.method == "POST":
        try:
            if not client:
                raise ValueError("GEMINI_API_KEY পাওয়া যায়নি।")
            files = request.files.getlist("files")
            saved = 0
            for file in files:
                if file.filename:
                    fname = file.filename.replace(" ", "_")
                    file.save(os.path.join(UPLOAD_FOLDER, fname))
                    saved += 1
            if saved == 0:
                raise ValueError("ফাইল সিলেক্ট করুন।")
            count = create_embeddings()
            success_msg = f"{saved} টি ফাইল, {count} টি ধারা তৈরি হয়েছে।"
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
    file_list = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    has_emb = os.path.exists(EMB_FILE)
    return render_template("admin.html", files=file_list, has_emb=has_emb, error=error_msg, success=success_msg)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        question = request.json.get("question","").strip()
        contexts = search(question, top_k=5)
        if not contexts:
            return jsonify({"answer":"দুঃখিত, এই তথ্যটি ডকুমেন্টে নেই।", "sources":[]})
        context_text = "\n\n".join([f"[{c['source']}] {c['text']}" for c in contexts])
        prompt = f"শুধু CONTEXT থেকে বাংলায় উত্তর দাও।\nCONTEXT:{context_text}\nপ্রশ্ন:{question}"
        gen_models = ["gemini-2.0-flash", "models/gemini-2.0-flash", "gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"]
        last_err = None
        response = None
        for gm in gen_models:
            try:
                response = client.models.generate_content(model=gm, contents=prompt)
                break
            except Exception as e:
                last_err = e
                print(f"Gen failed {gm}: {e}")
                continue
        if response is None:
            raise last_err
        return jsonify({"answer": response.text, "sources": contexts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {e}", "sources":[]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
