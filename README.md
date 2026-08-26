
# সমবায় আইন AI - NotebookLM Clone V2 (Gemini)

## সেটআপ - 3 মিনিটে
1. এই ফোল্ডারটি VPS / Render.com / Hostinger Python hosting এ আপলোড করুন
2. .env.example কে .env নামে rename করে ভেতরে আপনার Gemini API Key বসান
   GEMINI_API_KEY=AIza...
   Key পাবেন: https://aistudio.google.com/app/apikey
3. pip install -r requirements.txt
4. python app.py

## ব্যবহার
- /admin এ গিয়ে আপনার সমবায় আইনের PDF/TXT আপলোড করুন
- AI অটোমেটিক ধারা অনুযায়ী ভাগ করে Embedding বানাবে
- / তে গিয়ে প্রশ্ন করুন

## কেন এটা NotebookLM এর মতো?
- TF-IDF না, Semantic Embedding (text-embedding-004)
- ধারা অনুযায়ী Chunking
- Threshold 0.35 + Strict Prompt = Zero Hallucination
- প্রতিটি উত্তরে Source + Score

## এরপর Offline Qdrant ভার্সন
এই ভার্সন চলার পর আমি আপনাকে Qdrant + all-MiniLM-L6-v2 দিয়ে 100% ফ্রি ভার্সন বানিয়ে দেব, যাতে কোনো API Key লাগবে না।
