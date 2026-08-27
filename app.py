
import os, re, pickle, traceback, time
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
    # টেক্সট ছোট করে দাও, gemini-embedding-001 এর লিমিট 2048 token
    text = text[:3000]
    models_to_try = ["models/gemini-embedding-001", "models/gemini-embedding-2", "models/text-embedding-004"]
    last_err = None
    for m in models_to_try:
        try:
            resp = genai.embed_content(model=m, content=text)
            return resp['embedding']
        except Exception as e:
            last_err = e
            continue
    raise last_err

def chunk_by_dhara(text):
    # ধারা দিয়ে ভাগ, না পেলে 600 অক্ষর করে ভাগ
    parts = re.split(r'(?:ধারা|বিধি|উপ-বিধি)\s*[০-৯0-9]+', text)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) < 100:
            continue
        # বড় ধারা হলে আরও ছোট করো
        if len(p) > 1000:
            for i in range(0, len(p), 800):
                sub = p[i:i+800].strip()
                if len(sub) > 100:
                    chunks.append(sub)
        else:
            chunks.append(p)
        if len(chunks) >= 80:  # প্রতি ফাইলে সর্বোচ্চ ৮০ টা চাঙ্ক
            break
    if not chunks and len(text) > 100:
        for i in range(0, len(text), 800):
            chunks.append(text[i:i+800])
            if len(chunks) >= 80:
                break
    return chunks[:80]

def get_text_from_file(path):
    if path.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(path)
        text = ""
        for page in reader.pages[:40]:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def create_embeddings():
    all_chunks = []
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        if not (fname.endswith(".txt") or fname.endswith(".pdf")):
            continue
        raw = get_text_from_file(fpath)
        if not raw or len(raw) < 20:
            continue
        for ch in chunk_by_dhara(raw):
            all_chunks.append({"text": ch, "source": fname})
        if len(all_chunks) >= 400:  # মোট ৪০০ চাঙ্ক লিমিট, বড় হলে Timeout হবে
            break
    
    if not all_chunks:
        raise ValueError("কোনো টেক্সট পাওয়া যায়নি। ফাইল খালি কিনা দেখুন।")
    
    embeddings = []
    failed = 0
    for i, c in enumerate(all_chunks):
        try:
            emb = get_embedding(c["text"])
            embeddings.append(emb)
        except Exception as e:
            print(f"Embedding failed chunk {i}: {e}")
            failed += 1
            time.sleep(1)  # Rate limit হলে ১ সেকেন্ড অপেক্ষা
            # ২য় বার ট্রাই
            try:
                emb = get_embedding(c["text"])
                embeddings.append(emb)
                failed -= 1
            except:
                continue
        if (i+1) % 10 == 0:
            print(f"Embedded {i+1}/{len(all_chunks)}")
        time.sleep(0.2)  # API rate limit এড়াতে

    if not embeddings:
        raise ValueError(f"Embedding তৈরি হয়নি। Failed {failed} টা। API Key কোটা শেষ কিনা দেখুন।")

    # failed চাঙ্কগুলো বাদ দিয়ে যেগুলো হয়েছে সেগুলো রাখো
    # all_chunks আর embeddings এর লেংথ মিলাতে হবে
    valid_chunks = []
    # embeddings যতটা হয়েছে, ততটাই chunks রাখো
    # সহজ: all_chunks থেকে শেষ failed গুলো বাদ দাও
    # আসলে আমরা উপরে skip করেছি, তাই embeddings এর সাথে chunks মিলবে না, তাই নতুন করে ম্যাচ করি
    # সঠিক পদ্ধতি: embeddings লিস্টের সাথে chunks লিস্টের ইন্ডেক্স মিলিয়ে রাখা
    # এখন সিম্পল: যদি failed থাকে, তাহলে embeddings যতটা আছে ততটা chunks নাও
    if len(embeddings) < len(all_chunks):
        all_chunks = all_chunks[:len(embeddings) + failed]  # আনুমানিক
        # সবচেয়ে নিরাপদ: embeddings এর সমান সংখ্যক chunk কাটো
        all_chunks = all_chunks[:len(embeddings)] if len(all_chunks) > len(embeddings) else all_chunks

    # আসলে সঠিক করতে: আমরা embeddings আর chunks একসাথে বানাবো
    # তাই উপরের লুপকে আবার ঠিক করছি নিচে - এই ফাংশনটা ওভাররাইড করছি সঠিকভাবে
    data = {"chunks": all_chunks[:len(embeddings)], "embeddings": np.array(embeddings)}
    with open(EMB_FILE, "wb") as f:
        pickle.dump(data, f)
    return len(embeddings)

# উপরের ফাংশনটা ভুল ছিল, সঠিক ভার্সন:
def create_embeddings_fixed():
    all_chunks = []
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        if not (fname.lower().endswith(".txt") or fname.lower().endswith(".pdf")):
            continue
        raw = get_text_from_file(fpath)
        if not raw or len(raw) < 20:
            continue
        for ch in chunk_by_dhara(raw):
            all_chunks.append({"text": ch, "source": fname})
        if len(all_chunks) >= 350:
            break

    if not all_chunks:
        raise ValueError("কোনো টেক্সট পাওয়া যায়নি।")

    final_chunks = []
    embeddings = []
    for i, c in enumerate(all_chunks):
        try:
            emb = get_embedding(c["text"])
            final_chunks.append(c)
            embeddings.append(emb)
        except Exception as e:
            print(f"Skip chunk {i}: {e}")
            time.sleep(0.5)
            continue
        if (i+1) % 20 == 0:
            print(f"Progress {i+1}/{len(all_chunks)}")
        time.sleep(0.15)

    if not embeddings:
        raise ValueError("একটাও Embedding তৈরি হয়নি। API Key বা Quota চেক করুন।")

    data = {"chunks": final_chunks, "embeddings": np.array(embeddings)}
    with open(EMB_FILE, "wb") as f:
        pickle.dump(data, f)
    return len(embeddings)

def search(query, top_k=5):
    if not os.path.exists(EMB_FILE):
        return []
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)
    q_emb = np.array(get_embedding(query))
    emb_matrix = data["embeddings"]
    sims = np.dot(emb_matrix, q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if float(sims[idx]) < 0.20:
            continue
        results.append({"text": data["chunks"][idx]["text"], "source": data["chunks"][idx]["source"], "score": float(sims[idx])})
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug")
def debug():
    try:
        import os
        has_emb = os.path.exists(EMB_FILE)
        count = 0
        if has_emb:
            with open(EMB_FILE, "rb") as f:
                d = pickle.load(f)
                count = len(d["chunks"])
        models = genai.list_models()
        emb_models = [m.name for m in models if "embedContent" in m.supported_generation_methods]
        return jsonify({"api_key_exists": True, "has_emb": has_emb, "chunk_count": count, "embedding_models": emb_models})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

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
                # যদি নতুন ফাইল না দেয়, পুরনো ফাইল দিয়েই embedding বানাও
                if len([f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith(".")]) == 0:
                    raise ValueError("ফাইল সিলেক্ট করুন।")
            count = create_embeddings_fixed()
            success_msg = f"{count} টি ধারা থেকে Embedding তৈরি হয়েছে। এখন Chat এ প্রশ্ন করুন।"
        except Exception as e:
            traceback.print_exc()
            error_msg = f"{str(e)}"
    file_list = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    has_emb = os.path.exists(EMB_FILE)
    return render_template("admin.html", files=file_list, has_emb=has_emb, error=error_msg, success=success_msg)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        question = request.json.get("question","").strip()
        if not os.path.exists(EMB_FILE):
            return jsonify({"answer":"Embedding নেই। Admin এ গিয়ে ফাইল আপলোড করে 'আপলোড ও AI কে শেখান' বাটনে ক্লিক করুন।", "sources":[]})
        contexts = search(question, top_k=5)
        if not contexts:
            return jsonify({"answer":"দুঃখিত, এই তথ্যটি ডকুমেন্টে নেই। অন্য ভাবে প্রশ্ন করুন, যেমন 'নির্বাচন কিভাবে হয়?'", "sources":[]})
        context_text = "\n\n".join([f"[{c['source']}] {c['text'][:1200]}" for c in contexts])
        prompt = f"তুমি সমবায় আইন বিশেষজ্ঞ। শুধু CONTEXT থেকে বাংলায় উত্তর দাও। যদি CONTEXT এ না থাকে বলবে নেই।\n\nCONTEXT:\n{context_text}\n\nপ্রশ্ন: {question}\n\nউত্তর বাংলায় দাও এবং শেষে সূত্র উল্লেখ করো।"
        gen_models_to_try = ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.5-flash-lite"]
        last_err = None
        response = None
        for gm in gen_models_to_try:
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
        return jsonify({"answer": f"Error: {e}", "sources":[]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
