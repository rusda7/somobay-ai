'use client'
import { useState, useRef, useEffect } from 'react'
import { Mic, Send, Trash2, Plus, LogOut, User, Settings, Menu } from 'lucide-react'
type Message = { role: 'user' | 'ai', content: string, cached?: boolean }

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', content: 'আসসালামু আলাইকুম! আমি সমবায় আইন সহকারী।\n সমবায় সমিতি আইন ও বিধিমালা এবং সার্কুলার থেকেই শুধুমাত্র নির্ভুল উত্তর দেব।\n যেকোনো প্রশ্ন করুন:\n' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://somobay-ai-backend.onrender.com'

  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}) }, [messages, loading])

  const sendMessage = async () => {
    if(!input.trim()) return
    const userMsg: Message = { role: 'user', content: input }
    setMessages(m => [...m, userMsg])
    const q = input
    setInput('')
    setLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/chat`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ question: q, user_id: 'demo' }) })
      const data = await res.json()
      setMessages(m => [...m, { role: 'ai', content: data.answer, cached: data.cached }])
    } catch {
      setMessages(m => [...m, { role:'ai', content: 'সার্ভার ব্যস্ত, ১০ সেকেন্ড পর আবার চেষ্টা করুন।' }])
    }
    setLoading(false)
  }

  const startVoice = () => {
    const SR = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
    if(!SR) { alert('ভয়েস সাপোর্ট নেই'); return; }
    const rec = new SR(); rec.lang = 'bn-BD'; rec.onstart = ()=>{}; rec.onend = ()=>{}; rec.onresult = (e:any)=>setInput(e.results[0][0].transcript); rec.start();
  }

  return (
    <div className="flex h-screen bg-[#F5F7F6] overflow-hidden">
      {/* Sidebar */}
      <div className="w-[280px] bg-[#0B5C33] text-white flex flex-col p-3 hidden md:flex shrink-0">
        <div className="flex items-center gap-2 p-2 mb-4"><div className="w-8 h-8 bg-white rounded flex items-center justify-center text-[#0B5C33] font-bold">স</div><div><p className="font-bold text-sm">সমবায় অধিদপ্তর</p><p className="text-[11px] opacity-80">আইন সহকারী </p></div></div>
        <button onClick={()=>setMessages([messages[0]])} className="bg-white/15 hover:bg-white/25 p-2.5 rounded-lg flex items-center gap-2 mb-4 text-sm"><Plus size={16}/> নতুন চ্যাট</button>
        <div className="text-[11px] opacity-70 p-2"> </div>
        <div className="mt-auto p-2 text-[10px] opacity-50">© ২০২৬ সমবায় অধিদপ্তর</div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        {/* Header */}
        <div className="h-14 border-b flex items-center justify-between px-4 bg-white shrink-0"><div className="flex items-center gap-2"><span className="font-semibold text-sm">সমবায় আইন AI</span><span className="text-[10px] bg-green-100 text-green-800 px-2 py-0.5 rounded-full">ChatGPT Style - 6 Rules</span></div><div className="flex items-center gap-2"><Settings size={16}/><User size={18}/></div></div>
        
        {/* 05 - Fix cut: flex-1 + overflow-y-auto + min-h-0 + full scroll */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden min-h-0 p-3 md:p-6 space-y-4 bg-[#F9FBFA]" style={{scrollBehavior:'smooth'}}>
          {messages.map((m,i)=>(
            <div key={i} className={`w-full max-w-3xl ${m.role==='user' ? 'ml-auto flex justify-end' : 'mr-auto'}`}>
              <div className={`px-5 py-4 rounded-2xl shadow-sm break-words ${m.role==='user' ? 'bg-[#0B5C33] text-white rounded-br-sm max-w-[85%]' : 'bg-white border border-gray-100 rounded-bl-sm w-full'}`}>
                {/* 03 & 05 - No cut, no broken font */}
                <p className="whitespace-pre-wrap break-words leading-7 text-[14.5px] font-[Noto Sans Bengali] overflow-visible" style={{wordBreak:'break-word', overflowWrap:'anywhere'}}>
                  {m.content}
                </p>
                {m.cached && <span className="inline-block mt-2 text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full">⚡ Cached - Token Saved</span>}
              </div>
            </div>
          ))}
          {loading && <div className="max-w-3xl"><div className="bg-white border p-4 rounded-2xl"><p className="text-sm text-gray-500 animate-pulse">আপনার আপলোড করা ফাইল থেকে খুঁজছি...</p></div></div>}
          <div ref={bottomRef} />
        </div>

        {/* Input - fixed bottom */}
        <div className="p-3 md:p-4 bg-white border-t shrink-0">
          <div className="max-w-3xl mx-auto flex items-center gap-2 bg-gray-100 rounded-full px-3 py-2">
            <button onClick={startVoice} className="p-2.5 rounded-full bg-white hover:bg-gray-50 shrink-0"><Mic size={18}/></button>
            <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==='Enter' && sendMessage()} placeholder="প্রশ্ন লিখুন... (Enter চাপুন)" className="flex-1 bg-transparent outline-none text-[14px] min-w-0"/>
            <button onClick={sendMessage} disabled={loading} className="bg-[#0B5C33] hover:bg-[#094d2b] text-white p-2.5 rounded-full shrink-0 disabled:opacity-50"><Send size={18}/></button>
          </div>
          <p className="text-[10px] text-center text-gray-400 mt-2">শুধুমাত্র সমবায় সমিতি আইন ও বিধিমালা এবং সার্কুলার থেকে উত্তর দেয় • ভুল উত্তর দেয় না</p>
        </div>
      </div>
    </div>
  )
}
