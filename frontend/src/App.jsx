
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API_BASE_URL = 'http://127.0.0.1:8000'

const RISK_CONFIG = {
  RED: {
    label: 'High risk',
    badge: 'bg-red-100 text-red-700',
    border: 'border-red-200',
    dot: 'bg-red-500',
    description: 'Review this clause carefully.',
  },
  YELLOW: {
    label: 'Medium risk',
    badge: 'bg-amber-100 text-amber-800',
    border: 'border-amber-200',
    dot: 'bg-amber-500',
    description: 'Review the wording and conditions.',
  },
  GREEN: {
    label: 'Low risk',
    badge: 'bg-emerald-100 text-emerald-700',
    border: 'border-emerald-200',
    dot: 'bg-emerald-500',
    description: 'The classifier assigned a low-risk label.',
  },
  UNKNOWN: {
    label: 'Unclassified',
    badge: 'bg-slate-100 text-slate-700',
    border: 'border-slate-200',
    dot: 'bg-slate-400',
    description: 'No recognised risk label was returned.',
  },
}

const markdownClasses =
  'text-sm leading-7 text-slate-700 ' +
  '[&_h1]:mb-3 [&_h1]:text-xl [&_h1]:font-bold ' +
  '[&_h2]:mb-2 [&_h2]:mt-5 [&_h2]:text-lg [&_h2]:font-semibold ' +
  '[&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:font-semibold ' +
  '[&_p]:mb-3 [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-6 ' +
  '[&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-6 ' +
  '[&_li]:mb-1 [&_strong]:font-semibold [&_strong]:text-slate-900 ' +
  '[&_table]:my-4 [&_table]:w-full [&_table]:border-collapse ' +
  '[&_th]:border [&_th]:border-slate-300 [&_th]:bg-slate-100 ' +
  '[&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:font-semibold ' +
  '[&_td]:border [&_td]:border-slate-300 [&_td]:px-3 [&_td]:py-2 ' +
  '[&_blockquote]:border-l-4 [&_blockquote]:border-cyan-300 ' +
  '[&_blockquote]:pl-4 [&_blockquote]:italic'

function MarkdownContent({ children }) {
  return (
    <div className={markdownClasses}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {children || ''}
      </ReactMarkdown>
    </div>
  )
}

function formatRisk(value) {
  const key = String(value || 'UNKNOWN').toUpperCase()
  return RISK_CONFIG[key] ? key : 'UNKNOWN'
}

function getRiskItems(data, risk) {
  const keys = {
    RED: 'red_clauses',
    YELLOW: 'yellow_clauses',
    GREEN: 'green_clauses',
    UNKNOWN: 'unknown_clauses',
  }

  const items = data?.[keys[risk]]
  return Array.isArray(items) ? items : []
}

function getRiskCount(data, risk) {
  const directCount = data?.counts?.[risk]
  if (typeof directCount === 'number') return directCount
  return getRiskItems(data, risk).length
}

async function readResponse(response) {
  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    const detail =
      typeof data.detail === 'string'
        ? data.detail
        : JSON.stringify(data.detail ?? data)

    if (response.status === 401 || response.status === 403) {
      throw new Error(
        'Authentication failed. Log in again through Swagger and paste a fresh access token.',
      )
    }

    throw new Error(
      detail || `Request failed with status ${response.status}.`,
    )
  }

  return data
}

export default function App() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const [token, setToken] = useState('')
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [qaError, setQaError] = useState('')
  const [qaResult, setQaResult] = useState(null)

  const [activeView, setActiveView] = useState('dashboard')
  const [riskData, setRiskData] = useState(null)
  const [riskFilter, setRiskFilter] = useState('ALL')
  const [loadingRisks, setLoadingRisks] = useState(false)
  const [riskError, setRiskError] = useState('')

  const [explainingChunk, setExplainingChunk] = useState(null)
  const [explanations, setExplanations] = useState({})
  const [explainErrors, setExplainErrors] = useState({})

  function getAccessToken() {
    const enteredToken = token.trim()
      ? token.trim()
      : window.prompt('Paste your login access token')?.trim() || ''

    if (enteredToken) setToken(enteredToken)
    return enteredToken
  }

  async function handleUpload() {
    if (!file || uploading) return

    const activeToken = getAccessToken()

    if (!activeToken) {
      setError('A login token is required. Log in through Swagger first.')
      return
    }

    setUploading(true)
    setError('')
    setResult(null)
    setQaResult(null)
    setQaError('')
    setRiskData(null)
    setRiskError('')
    setExplanations({})
    setExplainErrors({})

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/contract/upload`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${activeToken}`,
        },
        body: formData,
      })

      const data = await readResponse(response)
      setResult(data)
      setQuestion('')
      setRiskFilter('ALL')
      setActiveView('dashboard')
    } catch (err) {
      setError(
        err instanceof TypeError
          ? 'Cannot connect to backend. Check that FastAPI is running and CORS is configured.'
          : err.message || 'Upload failed.',
      )
    } finally {
      setUploading(false)
    }
  }

  async function handleAskQuestion(event) {
    event.preventDefault()

    if (!result?.document_id) {
      setQaError('Upload a contract before asking questions.')
      return
    }

    if (!question.trim()) {
      setQaError('Please enter a question.')
      return
    }

    const activeToken = getAccessToken()

    if (!activeToken) {
      setQaError('A login token is required.')
      return
    }

    setAsking(true)
    setQaError('')
    setQaResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/contract/query`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${activeToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_id: result.document_id,
          question: question.trim(),
          top_k: 5,
        }),
      })

      const data = await readResponse(response)
      setQaResult(data)
    } catch (err) {
      setQaError(
        err instanceof TypeError
          ? 'Cannot connect to backend. Check FastAPI and CORS settings.'
          : err.message || 'Could not get an answer.',
      )
    } finally {
      setAsking(false)
    }
  }

  async function handleOpenRiskAnalysis() {
    setActiveView('risk')

    if (!result?.document_id) {
      setRiskData(null)
      setRiskError('Upload a contract first to view its risk analysis.')
      return
    }

    const activeToken = getAccessToken()

    if (!activeToken) {
      setRiskError('A login token is required. Log in through Swagger first.')
      return
    }

    setLoadingRisks(true)
    setRiskError('')

    try {
      const response = await fetch(
        `${API_BASE_URL}/contract/risk/${encodeURIComponent(result.document_id)}`,
        {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${activeToken}`,
          },
        },
      )

      const data = await readResponse(response)
      setRiskData(data)
    } catch (err) {
      setRiskError(
        err instanceof TypeError
          ? 'Cannot connect to backend. Check that FastAPI is running.'
          : err.message || 'Could not load risk analysis.',
      )
    } finally {
      setLoadingRisks(false)
    }
  }

  async function handleExplainRisk(item, risk, fallbackIndex) {
    const chunkIndex = item.chunk_index ?? fallbackIndex
    const explanationKey = `${risk}-${chunkIndex}`

    if (explanations[explanationKey]) {
      setExplanations((previous) => {
        const next = { ...previous }
        delete next[explanationKey]
        return next
      })
      return
    }

    const activeToken = getAccessToken()

    if (!activeToken) {
      setExplainErrors((previous) => ({
        ...previous,
        [explanationKey]: 'A login token is required.',
      }))
      return
    }

    if (!result?.document_id) {
      setExplainErrors((previous) => ({
        ...previous,
        [explanationKey]: 'Upload a contract first.',
      }))
      return
    }

    setExplainingChunk(explanationKey)
    setExplainErrors((previous) => ({
      ...previous,
      [explanationKey]: '',
    }))

    try {
      const response = await fetch(
        `${API_BASE_URL}/contract/risk/${encodeURIComponent(result.document_id)}/explain/${encodeURIComponent(chunkIndex)}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${activeToken}`,
          },
        },
      )

      const data = await readResponse(response)

      setExplanations((previous) => ({
        ...previous,
        [explanationKey]: data,
      }))
    } catch (err) {
      setExplainErrors((previous) => ({
        ...previous,
        [explanationKey]:
          err instanceof TypeError
            ? 'Cannot connect to backend.'
            : err.message || 'Could not explain this clause.',
      }))
    } finally {
      setExplainingChunk(null)
    }
  }

  const riskCounts = result?.risk_counts ?? {}

  const stats = [
    {
      label: 'Contracts Analyzed',
      value: result ? 1 : 0,
      icon: '📄',
    },
    {
      label: 'High Risk Clauses',
      value: riskCounts.RED ?? 0,
      icon: '🔴',
    },
    {
      label: 'Medium Risk Clauses',
      value: riskCounts.YELLOW ?? 0,
      icon: '🟡',
    },
    {
      label: 'Low Risk Clauses',
      value: riskCounts.GREEN ?? 0,
      icon: '🟢',
    },
  ]

  const availableRisks = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']

  const visibleRiskItems = (
    riskFilter === 'ALL'
      ? availableRisks.flatMap((risk) =>
          getRiskItems(riskData, risk).map((item) => ({
            ...item,
            risk_level: risk,
          })),
        )
      : getRiskItems(riskData, riskFilter).map((item) => ({
          ...item,
          risk_level: riskFilter,
        }))
  ).sort((a, b) => (a.chunk_index ?? 0) - (b.chunk_index ?? 0))

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 md:flex">
      <aside className="w-full bg-slate-950 p-6 text-white md:min-h-screen md:w-64 md:shrink-0">
        <h1 className="text-2xl font-bold tracking-tight">
          ContractSense<span className="text-cyan-400">AI</span>
        </h1>

        <p className="mt-2 text-sm text-slate-400">
          Intelligent Contract Analysis
        </p>

        <nav className="mt-10 space-y-2">
          <button
            type="button"
            onClick={() => setActiveView('dashboard')}
            className={`w-full rounded-xl px-4 py-3 text-left font-medium transition ${
              activeView === 'dashboard'
                ? 'bg-cyan-500/15 text-cyan-300'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            ▦ Dashboard
          </button>

          <button
            type="button"
            onClick={() => setActiveView('dashboard')}
            className="w-full rounded-xl px-4 py-3 text-left text-slate-300 transition hover:bg-slate-800"
          >
            ◫ My Contracts
          </button>

          <button
            type="button"
            onClick={handleOpenRiskAnalysis}
            className={`w-full rounded-xl px-4 py-3 text-left font-medium transition ${
              activeView === 'risk'
                ? 'bg-cyan-500/15 text-cyan-300'
                : 'text-slate-300 hover:bg-slate-800'
            }`}
          >
            ◈ Risk Analysis
          </button>

          <button
            type="button"
            onClick={() =>
              window.alert('Settings will be implemented in a later phase.')
            }
            className="w-full rounded-xl px-4 py-3 text-left text-slate-300 transition hover:bg-slate-800"
          >
            ⚙ Settings
          </button>
        </nav>

        <div className="mt-10 rounded-xl border border-slate-800 p-4">
          <p className="text-sm font-semibold">AI-powered protection</p>
          <p className="mt-2 text-xs leading-5 text-slate-400">
            Understand contract risks before you sign.
          </p>
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-5 md:p-10">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm text-slate-500">
              Workspace / {activeView === 'risk' ? 'Risk Analysis' : 'Overview'}
            </p>

            <h2 className="mt-2 text-3xl font-bold tracking-tight">
              {activeView === 'risk'
                ? 'Contract Risk Analysis'
                : 'Contract Dashboard'}
            </h2>

            <p className="mt-2 text-slate-500">
              {activeView === 'risk'
                ? 'Review clauses and ask AI to explain individual risks.'
                : 'Upload, analyze and understand your contracts.'}
            </p>
          </div>

          <div className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium">
            <span className="text-emerald-600">● Workspace ready</span>
          </div>
        </header>

        {activeView === 'dashboard' && (
          <>
            <section className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {stats.map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-slate-500">{stat.label}</p>
                    <span className="text-xl">{stat.icon}</span>
                  </div>
                  <p className="mt-4 text-3xl font-bold">{stat.value}</p>
                </div>
              ))}
            </section>

            <section className="mt-8 grid grid-cols-1 gap-6 xl:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm xl:col-span-2">
                <h3 className="text-xl font-semibold">Analyze a contract</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Select a PDF agreement to run AI-powered risk analysis.
                </p>

                <label className="mt-6 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center transition hover:border-cyan-500 hover:bg-cyan-50/50">
                  <span className="text-4xl">📄</span>
                  <span className="mt-4 break-all font-semibold">
                    {file ? file.name : 'Choose a contract PDF'}
                  </span>
                  <span className="mt-2 text-sm text-slate-500">
                    PDF files only · Select from your device
                  </span>
                  <input
                    type="file"
                    accept=".pdf,application/pdf"
                    className="sr-only"
                    disabled={uploading}
                    onChange={(event) => {
                      const selectedFile = event.target.files?.[0] ?? null
                      setFile(selectedFile)
                      setError('')
                      setResult(null)
                      setQaResult(null)
                      setQaError('')
                      setRiskData(null)
                      setRiskError('')
                      setExplanations({})
                      setExplainErrors({})
                    }}
                  />
                </label>

                <button
                  type="button"
                  disabled={!file || uploading}
                  onClick={handleUpload}
                  className="mt-5 w-full rounded-xl bg-slate-900 px-5 py-3 font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300 sm:w-auto"
                >
                  {uploading ? 'Uploading and analyzing...' : 'Upload & Analyze →'}
                </button>

                {error && (
                  <div
                    role="alert"
                    className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700"
                  >
                    <p className="font-semibold">Upload failed</p>
                    <p className="mt-1">{error}</p>
                  </div>
                )}

                {result && (
                  <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50/50 p-5">
                    <h4 className="font-semibold text-emerald-800">
                      ✓ Contract uploaded successfully
                    </h4>

                    <div className="mt-4 space-y-2 text-sm">
                      <p className="break-all">
                        <strong>File:</strong> {result.filename}
                      </p>
                      <p className="break-all">
                        <strong>Document ID:</strong> {result.document_id}
                      </p>
                      <p><strong>Pages:</strong> {result.page_count}</p>
                      <p><strong>Chunks:</strong> {result.chunk_count}</p>
                    </div>

                    <h5 className="mt-5 font-semibold">Risk breakdown</h5>
                    <div className="mt-3 flex flex-wrap gap-2 text-sm">
                      <span className="rounded-lg bg-red-100 px-3 py-2 font-medium text-red-700">
                        RED: {riskCounts.RED ?? 0}
                      </span>
                      <span className="rounded-lg bg-yellow-100 px-3 py-2 font-medium text-yellow-800">
                        YELLOW: {riskCounts.YELLOW ?? 0}
                      </span>
                      <span className="rounded-lg bg-green-100 px-3 py-2 font-medium text-green-700">
                        GREEN: {riskCounts.GREEN ?? 0}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={handleOpenRiskAnalysis}
                      className="mt-4 rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-700"
                    >
                      View detailed risk analysis →
                    </button>

                    {result.summary && (
                      <div className="mt-6 border-t border-emerald-200 pt-5">
                        <h5 className="mb-3 font-semibold">Contract summary</h5>
                        <MarkdownContent>{result.summary}</MarkdownContent>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-xl font-semibold">How it works</h3>
                <div className="mt-6 space-y-6">
                  {[
                    ['01', 'Upload contract', 'Select a PDF agreement.'],
                    ['02', 'Detect risks', 'Identify red, yellow and green clauses.'],
                    ['03', 'Ask questions', 'Get answers grounded in contract text.'],
                    ['04', 'Explain risks', 'Ask AI to explain an individual clause.'],
                  ].map(([number, title, description]) => (
                    <div key={number} className="flex gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-cyan-50 font-bold text-cyan-700">
                        {number}
                      </div>
                      <div>
                        <p className="font-semibold">{title}</p>
                        <p className="mt-1 text-sm leading-5 text-slate-500">
                          {description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-8 rounded-xl bg-amber-50 p-4 text-sm leading-6 text-amber-800">
                  AI-generated analysis is informational and should be reviewed
                  by a qualified legal professional.
                </div>
              </div>
            </section>

            <section className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h3 className="text-xl font-semibold">Ask your contract</h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Ask a question about the uploaded contract.
              </p>

              <form onSubmit={handleAskQuestion} className="mt-5">
                <label htmlFor="contract-question" className="mb-2 block text-sm font-medium">
                  Your question
                </label>
                <textarea
                  id="contract-question"
                  rows={3}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. What is the termination notice period?"
                  className="w-full rounded-xl border border-slate-300 p-3 outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                />

                <button
                  type="submit"
                  disabled={!result?.document_id || asking || !question.trim()}
                  className="mt-3 rounded-xl bg-cyan-700 px-5 py-3 font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {asking ? 'Finding answer...' : 'Ask AI →'}
                </button>
              </form>

              {qaError && (
                <div role="alert" className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                  {qaError}
                </div>
              )}

              {qaResult && (
                <div className="mt-5 rounded-xl border border-cyan-200 bg-cyan-50/50 p-5">
                  <p className="text-sm font-semibold text-cyan-900">
                    Question: {qaResult.question || question}
                  </p>
                  <h4 className="mt-4 font-semibold">AI answer</h4>
                  <div className="mt-2">
                    <MarkdownContent>{qaResult.answer}</MarkdownContent>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500">
                    <span>Chunks used: {qaResult.chunks_used ?? 0}</span>
                    <span>
                      Top similarity:{' '}
                      {typeof qaResult.top_similarity === 'number'
                        ? qaResult.top_similarity.toFixed(3)
                        : 'N/A'}
                    </span>
                  </div>

                  {Array.isArray(qaResult.retrieved_chunks) &&
                    qaResult.retrieved_chunks.length > 0 && (
                      <details className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
                        <summary className="cursor-pointer text-sm font-semibold">
                          View retrieved contract passages
                        </summary>
                        <div className="mt-4 space-y-4">
                          {qaResult.retrieved_chunks.map((chunk, index) => (
                            <div
                              key={chunk.chunk_index ?? index}
                              className="rounded-lg bg-slate-50 p-3 text-sm"
                            >
                              <p className="mb-2 font-semibold">
                                Passage {index + 1}
                              </p>
                              <p className="whitespace-pre-wrap text-slate-600">
                                {typeof chunk === 'string'
                                  ? chunk
                                  : chunk.text ?? chunk.content ?? JSON.stringify(chunk)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                </div>
              )}
            </section>
          </>
        )}

        {activeView === 'risk' && (
          <section className="mt-8">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <h3 className="text-xl font-semibold">Detailed risk analysis</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-500">
                    Review each classified clause and use Explain Risk to
                    understand why it may matter.
                  </p>
                  {result && (
                    <p className="mt-2 break-all text-xs text-slate-400">
                      Document: {result.filename} · ID: {result.document_id}
                    </p>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleOpenRiskAnalysis}
                  disabled={loadingRisks || !result?.document_id}
                  className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loadingRisks ? 'Refreshing...' : '↻ Refresh analysis'}
                </button>
              </div>

              {riskError && (
                <div role="alert" className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                  {riskError}
                </div>
              )}

              {loadingRisks && (
                <div className="mt-8 rounded-xl bg-slate-50 p-8 text-center">
                  <p className="font-semibold">Loading risk analysis...</p>
                  <p className="mt-2 text-sm text-slate-500">
                    Retrieving classified contract clauses.
                  </p>
                </div>
              )}

              {!loadingRisks && riskData && (
                <>
                  <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
                    {[
                      { key: 'RED', label: 'High risk', color: 'border-red-200 bg-red-50 text-red-700' },
                      { key: 'YELLOW', label: 'Medium risk', color: 'border-amber-200 bg-amber-50 text-amber-800' },
                      { key: 'GREEN', label: 'Low risk', color: 'border-emerald-200 bg-emerald-50 text-emerald-700' },
                      { key: 'UNKNOWN', label: 'Unclassified', color: 'border-slate-200 bg-slate-50 text-slate-700' },
                    ].map((item) => (
                      <div key={item.key} className={`rounded-xl border p-4 ${item.color}`}>
                        <p className="text-sm font-medium">{item.label}</p>
                        <p className="mt-2 text-2xl font-bold">
                          {getRiskCount(riskData, item.key)}
                        </p>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 flex flex-wrap gap-2">
                    {[
                      { key: 'ALL', label: 'All clauses' },
                      { key: 'RED', label: '🔴 High risk' },
                      { key: 'YELLOW', label: '🟡 Medium risk' },
                      { key: 'GREEN', label: '🟢 Low risk' },
                      { key: 'UNKNOWN', label: 'Unclassified' },
                    ].map((item) => (
                      <button
                        type="button"
                        key={item.key}
                        onClick={() => setRiskFilter(item.key)}
                        className={`rounded-full border px-4 py-2 text-sm font-medium transition ${
                          riskFilter === item.key
                            ? 'border-slate-900 bg-slate-900 text-white'
                            : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>

                  <div className="mt-5 flex items-center justify-between gap-3">
                    <h4 className="font-semibold">
                      {riskFilter === 'ALL'
                        ? 'All classified clauses'
                        : `${RISK_CONFIG[riskFilter]?.label || 'Risk'} clauses`}
                    </h4>
                    <span className="text-sm text-slate-500">
                      {visibleRiskItems.length} shown
                    </span>
                  </div>

                  {visibleRiskItems.length === 0 && (
                    <div className="mt-4 rounded-xl border border-dashed border-slate-300 p-8 text-center">
                      <p className="font-semibold">No clauses in this category</p>
                      <p className="mt-2 text-sm text-slate-500">
                        Try another filter or upload a different contract.
                      </p>
                    </div>
                  )}

                  <div className="mt-4 space-y-4">
                    {visibleRiskItems.map((item, index) => {
                      const risk = formatRisk(item.risk_level)
                      const config = RISK_CONFIG[risk]
                      const chunkIndex = item.chunk_index ?? index
                      const explanationKey = `${risk}-${chunkIndex}`
                      const explanation = explanations[explanationKey]
                      const explainError = explainErrors[explanationKey]
                      const isExplaining = explainingChunk === explanationKey

                      return (
                        <article
                          key={explanationKey}
                          className={`rounded-xl border bg-white p-5 ${config.border}`}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <span className={`h-2.5 w-2.5 rounded-full ${config.dot}`} />
                              <h5 className="font-semibold">
                                Clause chunk #{chunkIndex + 1}
                              </h5>
                            </div>

                            <span className={`rounded-full px-3 py-1 text-xs font-semibold ${config.badge}`}>
                              {config.label}
                            </span>
                          </div>

                          <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-700">
                            {item.text || 'No clause text was returned for this item.'}
                          </p>

                          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-3">
                            <p className="text-xs text-slate-500">
                              {config.description}
                            </p>
                            {typeof item.confidence === 'number' && (
                              <p className="text-xs text-slate-500">
                                Model confidence: {(item.confidence * 100).toFixed(1)}%
                              </p>
                            )}
                          </div>

                          <button
                            type="button"
                            onClick={() => handleExplainRisk(item, risk, index)}
                            disabled={isExplaining}
                            className="mt-4 rounded-xl bg-cyan-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            {isExplaining
                              ? 'Generating explanation...'
                              : explanation
                                ? 'Hide explanation ↑'
                                : '✨ Explain Risk'}
                          </button>

                          {explainError && (
                            <div role="alert" className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                              {explainError}
                            </div>
                          )}

                          {explanation && (
                            <div className="mt-4 rounded-xl border border-cyan-200 bg-cyan-50/60 p-4">
                              <h6 className="font-semibold text-cyan-950">
                                AI explanation
                              </h6>

                              {explanation.risk_level && (
                                <p className="mt-2 text-xs text-slate-500">
                                  Classifier label: {formatRisk(explanation.risk_level)}
                                </p>
                              )}

                              <div className="mt-3">
                                <MarkdownContent>
                                  {explanation.explanation ||
                                    explanation.analysis ||
                                    explanation.answer ||
                                    explanation.message ||
                                    'The API returned a response, but no explanation field was found. Check the endpoint response in Swagger.'}
                                </MarkdownContent>
                              </div>

                              {explanation.disclaimer && (
                                <p className="mt-3 border-t border-cyan-200 pt-3 text-xs leading-5 text-slate-600">
                                  {explanation.disclaimer}
                                </p>
                              )}
                            </div>
                          )}
                        </article>
                      )
                    })}
                  </div>

                  <div className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-900">
                    <strong>Important:</strong> Risk categories are predictions
                    from the project classifier, not definitive legal judgments.
                    AI explanations can also be imperfect. Review the original
                    contract and consult a qualified legal professional for
                    consequential decisions.
                  </div>
                </>
              )}

              {!loadingRisks && !riskData && !riskError && (
                <div className="mt-6 rounded-xl border border-dashed border-slate-300 p-8 text-center">
                  <p className="font-semibold">No risk analysis loaded yet</p>
                  <p className="mt-2 text-sm text-slate-500">
                    Upload a contract, then open Risk Analysis.
                  </p>
                </div>
              )}
            </div>
          </section>
        )}

        <footer className="mt-8 border-t border-slate-200 py-5 text-sm text-slate-400">
          ContractSenseAI · AI-assisted contract intelligence
        </footer>
      </main>
    </div>
  )
}
