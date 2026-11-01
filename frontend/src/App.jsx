import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Sun, Moon, ShieldCheck, Search, Network, BookOpen, Activity, ArrowUpRight, Sparkles, FileText, CircleHelp } from 'lucide-react'

const API = 'http://127.0.0.1:8000'
const samples = [
  { claim: 'The Earth orbits the Sun.', evidence: 'Earth travels around the Sun once approximately every 365 days.' },
  { claim: 'The capital of France is Berlin.', evidence: 'Paris is the capital and largest city of France.' },
  { claim: 'The university opened a new library in 2024.', evidence: 'The supplied passage discusses student examinations but gives no library opening date.' }
]

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('veritas-theme') || 'dark')
  const [claim, setClaim] = useState(samples[0].claim)
  const [evidence, setEvidence] = useState(samples[0].evidence)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('veritas-theme', theme)
  }, [theme])

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(d => setApiStatus(d.mode)).catch(() => setApiStatus('offline'))
  }, [])

  async function analyze() {
    setLoading(true); setResult(null)
    try {
      const response = await fetch(`${API}/api/analyze`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim, evidence })
      })
      if (!response.ok) throw new Error('The API could not process this request.')
      setResult(await response.json())
    } catch (e) {
      setResult({ error: e.message })
    } finally { setLoading(false) }
  }

  const labelColor = result?.label === 'SUPPORTED' ? 'green' : result?.label === 'REFUTED' ? 'red' : 'amber'

  return <div className="app-shell">
    <div className="ambient ambient-one"/><div className="ambient ambient-two"/>
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-mark"><ShieldCheck size={19}/></span><span>VERITAS<small>CLAIM & CONTEXT INTELLIGENCE</small></span></a>
      <nav><a href="#studio">Claim Studio</a><a href="#method">Methodology</a><a href="#about">About</a></nav>
      <div className="top-actions">
        <span className={`status-dot ${apiStatus === 'offline' ? 'off' : ''}`}/><span className="status-text">{apiStatus === 'trained_model' ? 'MODEL ONLINE' : apiStatus === 'demo_mode' ? 'DEMO MODE' : apiStatus === 'offline' ? 'API OFFLINE' : 'CONNECTING'}</span>
        <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle colour theme">{theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>}</button>
      </div>
    </header>

    <main>
      <section className="hero">
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .7 }}>
          <div className="eyebrow"><span/> AN EVIDENCE-AWARE NLP RESEARCH PROJECT</div>
          <h1>Truth deserves<br/><em>to be examined.</em></h1>
          <p className="hero-copy">A neural approach to understanding claims, evaluating evidence, and revealing the relationships hidden within language.</p>
          <a className="text-link" href="#studio">Enter the Claim Studio <ArrowUpRight size={16}/></a>
        </motion.div>
        <motion.div className="hero-art" initial={{ opacity: 0, scale: .94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .9 }}>
          <div className="orbit orbit-a"/><div className="orbit orbit-b"/><div className="core"><ShieldCheck size={39}/></div>
          <span className="orbit-label label-a">CLAIM</span><span className="orbit-label label-b">EVIDENCE</span><span className="orbit-label label-c">CONTEXT</span>
          <div className="art-caption">INTERROGATE THE RELATIONSHIP<br/><b>NOT JUST THE STATEMENT</b></div>
        </motion.div>
      </section>

      <section className="metrics-strip">
        <div><span className="metric-icon"><Network size={17}/></span><b>Context-aware</b><small>Claim–evidence modelling</small></div>
        <div><span className="metric-icon"><Activity size={17}/></span><b>Neural analysis</b><small>BiLSTM + attention</small></div>
        <div><span className="metric-icon"><BookOpen size={17}/></span><b>Explainable</b><small>Evidence-led outputs</small></div>
      </section>

      <section className="studio-section" id="studio">
        <div className="section-heading"><div><div className="eyebrow">01 / INTERACTIVE WORKSPACE</div><h2>The Claim Studio</h2><p>Examine a statement against a passage of evidence supplied by you.</p></div><span className="section-index">VERITAS / 001</span></div>
        <div className="studio-grid">
          <div className="input-panel panel">
            <div className="panel-title"><span><FileText size={17}/> CLAIM & SOURCE</span><span className="tiny-label">INPUT</span></div>
            <label htmlFor="claim">Statement to examine</label>
            <textarea id="claim" value={claim} onChange={e => setClaim(e.target.value)} placeholder="Enter a factual claim..." />
            <label htmlFor="evidence">Evidence passage</label>
            <textarea id="evidence" className="evidence-input" value={evidence} onChange={e => setEvidence(e.target.value)} placeholder="Paste the relevant evidence passage..." />
            <div className="sample-row"><span>TRY A SAMPLE</span>{samples.map((s, i) => <button key={i} onClick={() => {setClaim(s.claim);setEvidence(s.evidence);setResult(null)}}>{String(i+1).padStart(2,'0')}</button>)}</div>
            <button className="analyze-button" onClick={analyze} disabled={loading || claim.trim().length < 8 || evidence.trim().length < 8}>{loading ? 'Examining relationship…' : <><Search size={17}/> Analyse claim & evidence <ArrowUpRight size={17}/></>}</button>
            <p className="privacy-note"><ShieldCheck size={13}/> Text is processed by your locally running project API.</p>
          </div>
          <div className="result-panel panel">
            <div className="panel-title"><span><Sparkles size={17}/> ANALYSIS REPORT</span><span className="tiny-label">OUTPUT</span></div>
            {!result && <div className="empty-state"><div className="empty-orb"><CircleHelp size={29}/></div><h3>Awaiting examination</h3><p>Your claim–evidence relationship will appear here, alongside the model's operating mode and interpretation.</p><div className="empty-rule"/></div>}
            {result?.error && <div className="error-state"><h3>Connection unavailable</h3><p>{result.error}</p><small>Start the FastAPI backend and try again.</small></div>}
            {result && !result.error && <div className="report">
              <div className="verdict-label">PREDICTED RELATIONSHIP</div>
              <div className={`verdict ${labelColor}`}>{result.label.replaceAll('_',' ')}</div>
              <div className="report-meta"><span>{result.mode === 'trained_model' ? 'TRAINED MODEL' : 'DEMO MODE'}</span>{result.confidence !== null && <span>CONFIDENCE {(result.confidence*100).toFixed(1)}%</span>}</div>
              <p className="report-summary">{result.summary}</p>
              <div className="explanation-box"><b>INTERPRETATION</b><p>{result.explanation}</p></div>
              {result.mode === 'demo_mode' && <div className="demo-warning">The neural model has not been trained yet. Follow the README training instructions to create a checkpoint.</div>}
            </div>}
            <div className="report-foot"><span>VERITAS RESEARCH INTERFACE</span><span>01 — 03</span></div>
          </div>
        </div>
      </section>

      <section className="method-section" id="method">
        <div className="section-heading"><div><div className="eyebrow">02 / UNDER THE SURFACE</div><h2>A neural reading of evidence.</h2><p>The research architecture is designed to make core deep-learning concepts inspectable.</p></div></div>
        <div className="method-grid">
          <article className="method-card"><span>01</span><Network/><h3>Sequence learning</h3><p>BiLSTM encodes the claim and evidence as contextual token sequences.</p></article>
          <article className="method-card"><span>02</span><Sparkles/><h3>Attention mechanism</h3><p>Learned attention weights create a view into the model's sequence representation.</p></article>
          <article className="method-card"><span>03</span><Activity/><h3>Neural classification</h3><p>An MLP predicts supported, refuted, or not-enough-information relationships.</p></article>
        </div>
      </section>

      <section className="disclaimer" id="about"><div className="disclaimer-icon"><ShieldCheck/></div><div><b>A note on responsible interpretation</b><p>VERITAS classifies the relationship between text passages supplied to it. It does not independently establish real-world truth, authenticate sources, or replace expert fact-checking. A model prediction is not proof.</p></div></section>
    </main>
    <footer><a className="footer-brand" href="#">VERITAS</a><span>AN NLP + DEEP LEARNING RESEARCH PROJECT</span><span>BUILT FOR INQUIRY, NOT ASSUMPTION.</span></footer>
  </div>
}
