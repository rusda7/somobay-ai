import os, re, pickle, traceback
import numpy as np
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import PyPDF2
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
EMB_FILE = "embeddings.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_embedding(text):
    # পুরনো stable library
    resp = genai.embed_content(model="models/text-embedding-004", content=text)
    return resp['embedding']

def chunk_by_dhara(text):
    parts = re.split(r'(?:ধারা|বিধি|উপ-বিধি)\s*[০-৯0-9]+', text)
    chunks = [p.strip() for p in parts if len(p.strip()) > 50]
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
        if not os.path.isfile(fpath) or fname.startswith("."): continue
        raw = get_text_from_file(fpath)
        if not raw or len(raw) < 20: continue
        for ch in chunk_by_dhara(raw):
            all_chunks.append({"text": ch, "source": fname})
    if not all_chunks:
        raise ValueError("কোনো টেক্সট পাওয়া যায়নি।")
    embeddings = []
    for i, c in enumerate(all_chunks):
        emb = get_embedding(c["text"])
        embeddings.append(emb)
        print(f"Embedded {i+1}/{len(all_chunks)}")
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
        if float(sims[idx]) < 0.25: continue
        results.append({"text": data["chunks"][idx]["text"], "source": data["chunks"][idx]["source"], "score": float(sims[idx])})
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug")
def debug():
    try:
        models = genai.list_models()
        names = [m.name for m in models if 'embedContent' in m.supported_generation_methods or 'generateContent' in m.supported_generation_methods]
        return jsonify({"api_key_exists": bool(api_key), "models": names[:30], "library": "google-generativeai 0.8.3"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "api_key_exists": bool(api_key)})

@app.route("/admin", methods=["GET","POST"])
def admin():
    error_msg = None
    success_msg = None
    if request.method == "POST":
        try:
            if not api_key:
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
        prompt = f"তুমি সমবায় আইন বিশেষজ্ঞ। শুধু নিচের CONTEXT থেকে বাংলায় উত্তর দাও। যদি CONTEXT এ না থাকে, বলবে নেই।\n\nCONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text, "sources": contexts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {e}", "sources":[]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
