
import os, re, pickle, traceback
from flask import Flask, request, render_template, jsonify, redirect, url_for
from dotenv import load_dotenv
import PyPDF2
import google.generativeai as genai
from collections import Counter
import math

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_text_from_file(path):
    if path.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages[:30]:  # কম পেজ
                t = page.extract_text() or ""
                text += t + "\n"
            return text
        except Exception as e:
            print(f"PDF error {path}: {e}")
            return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_smart(text, source):
    chunks = []
    pattern = r'(?=(?:ধারা|বিধি|উপ-বিধি|পরিচ্ছেদ|অধ্যায়)\s*[০-৯0-9]+)'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) < 100:
            continue
        if len(part) <= 1000:
            chunks.append({"text": part, "source": source})
        else:
            for i in range(0, len(part), 700):
                sub = part[i:i+900].strip()
                if len(sub) > 100:
                    chunks.append({"text": sub, "source": source})
        if len(chunks) >= 80:
            break
    if not chunks:
        for i in range(0, len(text), 700):
            sub = text[i:i+900].strip()
            if len(sub) > 100:
                chunks.append({"text": sub, "source": source})
            if len(chunks) >= 80:
                break
    return chunks[:80]

def build_index():
    all_chunks = []
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        if not fname.lower().endswith((".txt", ".pdf")):
            continue
        raw = get_text_from_file(fpath)
        if len(raw) < 20:
            continue
        all_chunks.extend(chunk_smart(raw, fname))
        if len(all_chunks) >= 300:  # memory save
            break
    
    index_data = {
        "chunks": all_chunks,
        "doc_freq": Counter(),
        "total_docs": len(all_chunks)
    }
    for ch in all_chunks:
        words = set(re.findall(r'[\u0980-\u09FF\w]+', ch["text"].lower()))
        for w in words:
            index_data["doc_freq"][w] += 1

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index_data, f)
    return len(all_chunks)

def search_fast(query, top_k=3):  # শুধু 3 টা context - memory save
    if not os.path.exists(INDEX_FILE):
        return []
    with open(INDEX_FILE, "rb") as f:
        index_data = pickle.load(f)
    
    query_words = re.findall(r'[\u0980-\u09FF\w]+', query.lower())
    if not query_words:
        return []
    
    scores = []
    N = index_data["total_docs"]
    for idx, ch in enumerate(index_data["chunks"]):
        text_lower = ch["text"].lower()
        score = 0
        for qw in query_words:
            if qw in text_lower:
                tf = text_lower.count(qw)
                df = index_data["doc_freq"].get(qw, 1)
                idf = math.log(N / df + 1)
                score += tf * idf
        if score > 0:
            scores.append((score, idx))
    
    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        results.append({
            "text": index_data["chunks"][idx]["text"],
            "source": index_data["chunks"][idx]["source"],
            "score": score
        })
    if not results and index_data["chunks"]:
        results = [{"text": index_data["chunks"][0]["text"], "source": index_data["chunks"][0]["source"], "score": 0}]
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug")
def debug():
    has_index = os.path.exists(INDEX_FILE)
    count = 0
    if has_index:
        try:
            with open(INDEX_FILE, "rb") as f:
                d = pickle.load(f)
                count = len(d["chunks"])
        except:
            pass
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f)) and not f.startswith(".")]
    return jsonify({"has_index": has_index, "chunk_count": count, "files": files})

@app.route("/admin", methods=["GET","POST"])
def admin():
    if request.method == "POST":
        files = request.files.getlist("files")
        saved = 0
        for file in files:
            if file.filename:
                fname = file.filename.replace(" ", "_")
                file.save(os.path.join(UPLOAD_FOLDER, fname))
                saved += 1
        try:
            count = build_index()
            return redirect(url_for('admin', success=f"{saved} ফাইল, {count} ধারা Index (Fast)। এখন Chat করুন।"))
        except Exception as e:
            traceback.print_exc()
            return redirect(url_for('admin', error=str(e)[:200]))

    file_list = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    has_index = os.path.exists(INDEX_FILE)
    success = request.args.get("success")
    error = request.args.get("error")
    return render_template("admin.html", files=file_list, has_emb=has_index, has_index=has_index, success=success, error=error)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        question = request.json.get("question","").strip()
        if not os.path.exists(INDEX_FILE):
            return jsonify({"answer": "Index নেই। Admin এ ফাইল আপলোড করুন।", "sources": []})
        
        contexts = search_fast(question, top_k=3)
        if not contexts:
            return jsonify({"answer": "দুঃখিত, প্রাসঙ্গিক তথ্য পাওয়া যায়নি।", "sources": []})

        # memory save: context ছোট করো
        context_text = "\n".join([f"{c['text'][:800]}" for c in contexts])
        prompt = f"CONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}\n\nবাংলায় সংক্ষেপে উত্তর দাও।"

        models_try = ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.5-flash-lite"]
        last_err = None
        resp_text = None
        for m in models_try:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content(prompt)
                resp_text = response.text
                break
            except Exception as e:
                last_err = e
                continue
        if resp_text is None:
            raise last_err

        return jsonify({"answer": resp_text, "sources": contexts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {e}", "sources": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
