
import os, re, pickle, traceback, json
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
            for page in reader.pages[:60]:
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
    # ধারা/বিধি দিয়ে ভাগ + overlap
    pattern = r'(?=(?:ধারা|বিধি|উপ-বিধি|পরিচ্ছেদ|অধ্যায়)\s*[০-৯0-9]+)'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) < 100:
            continue
        if len(part) <= 1200:
            chunks.append({"text": part, "source": source})
        else:
            for i in range(0, len(part), 800):
                sub = part[i:i+1100].strip()
                if len(sub) > 100:
                    chunks.append({"text": sub, "source": source})
        if len(chunks) >= 150:
            break
    if not chunks:
        for i in range(0, len(text), 800):
            sub = text[i:i+1100].strip()
            if len(sub) > 100:
                chunks.append({"text": sub, "source": source})
            if len(chunks) >= 150:
                break
    return chunks[:150]

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
        if len(all_chunks) >= 800:
            break
    
    # TF-IDF index for fast search (no embedding API needed)
    # প্রতি chunk এর word count
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

def search_fast(query, top_k=5):
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
        # keyword match + TF-IDF
        for qw in query_words:
            if qw in text_lower:
                tf = text_lower.count(qw)
                df = index_data["doc_freq"].get(qw, 1)
                idf = math.log(N / df + 1)
                score += tf * idf
                # বোনাস: শিরোনামে থাকলে
                if qw in ch["text"][:200].lower():
                    score += 2
        # নির্বাচন => ভোট, ভোটার, কমিটি
        if "নির্বাচন" in query:
            if any(k in text_lower for k in ["ভোট", "নির্বাচন", "প্রার্থী", "ব্যালট"]):
                score += 1
        if "ঋণ" in query and "ঋণ" in text_lower:
            score += 2
        
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
    # যদি একটাও না পায়, প্রথম 2 টা দেখাও
    if not results and index_data["chunks"]:
        results = [
            {"text": index_data["chunks"][0]["text"], "source": index_data["chunks"][0]["source"], "score": 0},
            {"text": index_data["chunks"][1]["text"] if len(index_data["chunks"])>1 else "", "source": index_data["chunks"][1]["source"] if len(index_data["chunks"])>1 else "", "score": 0}
        ]
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
    return jsonify({"has_index": has_index, "chunk_count": count, "files": files, "index_path": INDEX_FILE})

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
        # instant indexing - no embedding API, 1 second এ শেষ
        try:
            count = build_index()
            # redirect to avoid resend
            return redirect(url_for('admin', success=f"{saved} টি ফাইল, {count} টি ধারা Index করা হয়েছে। এখন Chat করুন।"))
        except Exception as e:
            traceback.print_exc()
            return redirect(url_for('admin', error=str(e)))

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
            return jsonify({"answer": "Index নেই। Admin এ গিয়ে ফাইল আপলোড করুন।", "sources": []})
        
        contexts = search_fast(question, top_k=5)
        if not contexts:
            return jsonify({"answer": "দুঃখিত, এই তথ্যটি ডকুমেন্টে নেই।", "sources": []})

        context_text = "\n\n".join([f"[{c['source']}] {c['text'][:1300]}" for c in contexts])
        prompt = f"তুমি বাংলাদেশের সমবায় আইন বিশেষজ্ঞ। নিচের CONTEXT থেকে প্রশ্নের উত্তর বাংলায় দাও। CONTEXT এ সরাসরি না থাকলেও সংশ্লিষ্ট তথ্য দিয়ে সাহায্য করো।\n\nCONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}\n\nউত্তর বাংলায় বিস্তারিত দাও এবং শেষে সূত্র (যেমন [ain_2001.txt]) উল্লেখ করো।"

        # শুধু 1 টা API কল - তাই দ্রুত (2-3 সেকেন্ড)
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
