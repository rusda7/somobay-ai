
import os, re, pickle, traceback, time, json, threading
import numpy as np
from flask import Flask, request, render_template, jsonify, Response
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
PROGRESS_FILE = "progress.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress_state = {"status": "idle", "total": 0, "done": 0, "message": ""}

def save_progress():
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress_state, f, ensure_ascii=False)
    except:
        pass

def get_embedding(text):
    text = text[:3000]
    models_to_try = ["models/gemini-embedding-001", "models/gemini-embedding-2", "models/text-embedding-004"]
    last_err = None
    for m in models_to_try:
        try:
            resp = genai.embed_content(model=m, content=text)
            return resp['embedding']
        except Exception as e:
            last_err = e
            # quota error হলে 2 সেকেন্ড wait
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(2)
            continue
    raise last_err

def chunk_by_dhara_smart(text):
    # 800 char with 150 overlap - Best for Bengali law
    chunks = []
    # প্রথমে ধারা দিয়ে ভাগ
    pattern = r'(?=(?:ধারা|বিধি|উপ-বিধি|পরিচ্ছেদ|অধ্যায়)\s*[০-৯0-9]+)'
    parts = re.split(pattern, text)
    for part in parts:
        part = part.strip()
        if len(part) < 80:
            continue
        if len(part) <= 1000:
            chunks.append(part)
        else:
            # বড় ধারা হলে overlap দিয়ে ভাগ
            for i in range(0, len(part), 700):
                sub = part[i:i+900].strip()
                if len(sub) > 80:
                    chunks.append(sub)
        if len(chunks) >= 100:
            break
    if not chunks:
        # fallback: sliding window
        for i in range(0, len(text), 700):
            sub = text[i:i+900].strip()
            if len(sub) > 80:
                chunks.append(sub)
            if len(chunks) >= 100:
                break
    return chunks[:100]

def get_text_from_file(path):
    if path.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages[:50]:
                t = page.extract_text() or ""
                text += t + "\n"
            return text
        except Exception as e:
            print(f"PDF read error {path}: {e}")
            return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def expand_query_bangla(q):
    # বাংলা query expansion - নির্বাচন => নির্বাচন, ভোট, নির্বাচন কমিটি
    expansions = {
        "নির্বাচন": "নির্বাচন ভোট নির্বাচন কমিটি নির্বাচনী",
        "সমিতি": "সমিতি সমবায় সমিতি সদস্য",
        "ঋণ": "ঋণ ঋণদান সুদ ঋণ খেলাপি",
        "অডিট": "অডিট নিরীক্ষা হিসাব",
        "বিধি": "বিধি নিয়ম বিধান",
    }
    extra = ""
    for key, val in expansions.items():
        if key in q:
            extra += " " + val
    return q + extra

def create_embeddings_background():
    global progress_state
    try:
        progress_state = {"status": "running", "total": 0, "done": 0, "message": "ফাইল পড়া হচ্ছে..."}
        save_progress()

        all_chunks = []
        for fname in os.listdir(UPLOAD_FOLDER):
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if not os.path.isfile(fpath) or fname.startswith("."):
                continue
            if not fname.lower().endswith((".txt", ".pdf")):
                continue
            raw = get_text_from_file(fpath)
            if not raw or len(raw) < 20:
                continue
            for ch in chunk_by_dhara_smart(raw):
                all_chunks.append({"text": ch, "source": fname})
            if len(all_chunks) >= 500:
                break

        if not all_chunks:
            progress_state = {"status": "error", "total": 0, "done": 0, "message": "কোনো টেক্সট পাওয়া যায়নি"}
            save_progress()
            return

        progress_state["total"] = len(all_chunks)
        progress_state["message"] = f"{len(all_chunks)} টি ধারা পাওয়া গেছে, Embedding শুরু..."
        save_progress()

        final_chunks = []
        embeddings = []
        for i, c in enumerate(all_chunks):
            try:
                emb = get_embedding(c["text"])
                final_chunks.append(c)
                embeddings.append(emb)
            except Exception as e:
                print(f"Skip {i}: {e}")
                # quota হলে একটু বেশি wait
                time.sleep(0.8)
                continue
            
            progress_state["done"] = i+1
            progress_state["message"] = f"Embedding {i+1}/{len(all_chunks)}"
            if (i+1) % 10 == 0:
                save_progress()
            time.sleep(0.12)  # rate limit

        if not embeddings:
            progress_state = {"status": "error", "total": len(all_chunks), "done": 0, "message": "Embedding ব্যর্থ, API Quota চেক করুন"}
            save_progress()
            return

        data = {"chunks": final_chunks, "embeddings": np.array(embeddings)}
        with open(EMB_FILE, "wb") as f:
            pickle.dump(data, f)

        progress_state = {"status": "done", "total": len(all_chunks), "done": len(embeddings), "message": f"{len(embeddings)} টি ধারা শেখানো সম্পন্ন!"}
        save_progress()

    except Exception as e:
        traceback.print_exc()
        progress_state = {"status": "error", "total": 0, "done": 0, "message": f"Error: {str(e)[:200]}"}
        save_progress()

def search(query, top_k=6):
    if not os.path.exists(EMB_FILE):
        return []
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)
    
    # query expansion
    expanded = expand_query_bangla(query)
    q_emb = np.array(get_embedding(expanded))
    emb_matrix = data["embeddings"]
    sims = np.dot(emb_matrix, q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        score = float(sims[idx])
        if score < 0.15:  # threshold কমানো হলো
            continue
        results.append({"text": data["chunks"][idx]["text"], "source": data["chunks"][idx]["source"], "score": score})
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/progress")
def api_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except:
            pass
    return jsonify(progress_state)

@app.route("/debug")
def debug():
    try:
        has_emb = os.path.exists(EMB_FILE)
        count = 0
        if has_emb:
            with open(EMB_FILE, "rb") as f:
                d = pickle.load(f)
                count = len(d["chunks"])
        return jsonify({"has_emb": has_emb, "chunk_count": count, "progress": progress_state})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/admin", methods=["GET","POST"])
def admin():
    global progress_state
    error_msg = None
    success_msg = None
    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            saved = 0
            for file in files:
                if file.filename:
                    fname = file.filename.replace(" ", "_")
                    file.save(os.path.join(UPLOAD_FOLDER, fname))
                    saved += 1
            # background thread এ embedding শুরু
            if saved > 0 or len([f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith(".")]) > 0:
                progress_state = {"status": "running", "total": 0, "done": 0, "message": "শুরু হচ্ছে..."}
                save_progress()
                thread = threading.Thread(target=create_embeddings_background, daemon=True)
                thread.start()
                success_msg = f"{saved} টি ফাইল আপলোড হয়েছে। Background এ AI শিখছে, Progress নিচে দেখুন।"
            else:
                raise ValueError("ফাইল সিলেক্ট করুন।")
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
        if not os.path.exists(EMB_FILE):
            return jsonify({"answer": "Embedding নেই। Admin এ গিয়ে ফাইল আপলোড করুন এবং 'আপলোড ও AI কে শেখান' এ ক্লিক করুন।", "sources": []})
        contexts = search(question, top_k=6)
        if not contexts:
            # threshold কমিয়ে আবার চেষ্টা, না পেলে সবচেয়ে কাছের 2 টা দেখাও
            with open(EMB_FILE, "rb") as f:
                data = pickle.load(f)
            expanded = expand_query_bangla(question)
            q_emb = np.array(get_embedding(expanded))
            emb_matrix = data["embeddings"]
            sims = np.dot(emb_matrix, q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9)
            top_idx = np.argsort(sims)[::-1][:2]
            contexts = [{"text": data["chunks"][i]["text"], "source": data["chunks"][i]["source"], "score": float(sims[i])} for i in top_idx]

        context_text = "\n\n".join([f"[{c['source']}] {c['text'][:1200]}" for c in contexts])
        prompt = f"তুমি বাংলাদেশের সমবায় আইন বিশেষজ্ঞ। CONTEXT থেকে প্রশ্নের উত্তর বাংলায় বিস্তারিত দাও। CONTEXT এ সরাসরি না থাকলেও সংশ্লিষ্ট তথ্য থেকে অনুমান করে সাহায্য করো। উত্তর শেষে সূত্র উল্লেখ করো।\n\nCONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}"

        gen_models = ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.5-flash-lite"]
        last_err = None
        response = None
        for gm in gen_models:
            try:
                model = genai.GenerativeModel(gm)
                response = model.generate_content(prompt)
                break
            except Exception as e:
                last_err = e
                continue
        if response is None:
            raise last_err
        return jsonify({"answer": response.text, "sources": contexts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {e}", "sources": []})

@app.route("/api/chat_stream", methods=["POST"])
def chat_stream():
    # streaming version - চিন্তা করতেই থাকে সমস্যা দূর করবে
    def generate():
        try:
            data = request.get_json()
            question = data.get("question","").strip()
            if not os.path.exists(EMB_FILE):
                yield f"data: {json.dumps({'token': 'Embedding নেই। Admin এ ফাইল আপলোড করুন।'})}\n\n"
                return
            contexts = search(question, top_k=5)
            if not contexts:
                yield f"data: {json.dumps({'token': 'দুঃখিত, প্রাসঙ্গিক তথ্য খুঁজে পাওয়া যায়নি। অন্য ভাবে প্রশ্ন করুন।'})}\n\n"
                return
            context_text = "\n\n".join([f"[{c['source']}] {c['text'][:1000]}" for c in contexts])
            prompt = f"CONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}\n\nবাংলায় সংক্ষেপে উত্তর দাও।"
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                    time.sleep(0.02)
            yield f"data: {json.dumps({'done': True, 'sources': contexts})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {e}'})}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
