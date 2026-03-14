"use client";
import { useState, useRef, useEffect } from "react";
import { streamQuery, StreamEvent, Chunk, Source } from "@/lib/api";
import { Send, Loader2, Sparkles, ChevronDown, ChevronUp, BookOpen } from "lucide-react";

interface HistoryItem {
    id: string;
    question: string;
    answer: string;
    sources: Source[];
    chunks: Chunk[];
}

function CitationBadge({ doc, page }: Source) {
    const short = doc.length > 22 ? doc.slice(0, 20) + "…" : doc;
    return (
        <span
            title={`${doc} — page ${page}`}
            style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "3px 9px", borderRadius: 999, fontSize: 11, fontWeight: 600,
                background: "rgba(91,110,245,0.1)", color: "var(--accent-light)",
                border: "1px solid rgba(91,110,245,0.25)", cursor: "default",
                whiteSpace: "nowrap",
            }}
        >
            <BookOpen size={10} strokeWidth={2} />
            {short} p.{page}
        </span>
    );
}

function ChunkCard({ chunk, idx }: { chunk: Chunk; idx: number }) {
    const [expanded, setExpanded] = useState(false);
    return (
        <div
            className="fade-in"
            style={{
                background: "var(--bg-card)", border: "1px solid var(--border)",
                borderRadius: 10, padding: "10px 14px", fontSize: 12,
                animationDelay: `${idx * 0.05}s`,
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <CitationBadge doc={chunk.doc} page={chunk.page} />
                <button
                    onClick={() => setExpanded((e) => !e)}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: 2 }}
                >
                    {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>
            </div>
            <p
                style={{
                    color: "var(--text-secondary)", lineHeight: 1.6, margin: 0,
                    overflow: "hidden",
                    maxHeight: expanded ? "none" : "3.2em",
                    display: "-webkit-box",
                    WebkitLineClamp: expanded ? undefined : 2,
                    WebkitBoxOrient: "vertical" as const,
                }}
            >
                {chunk.text}
            </p>
        </div>
    );
}

function AnswerBlock({ item }: { item: HistoryItem }) {
    const [showChunks, setShowChunks] = useState(false);
    return (
        <div className="fade-in" style={{ marginBottom: 28 }}>
            {/* Question */}
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
                <div
                    style={{
                        maxWidth: "75%", padding: "10px 16px", borderRadius: "18px 18px 4px 18px",
                        background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                        color: "white", fontSize: 14, lineHeight: 1.5, fontWeight: 500,
                    }}
                >
                    {item.question}
                </div>
            </div>

            {/* Answer */}
            <div
                style={{
                    maxWidth: "85%", padding: "16px 20px",
                    background: "var(--bg-card)", border: "1px solid var(--border)",
                    borderRadius: "4px 18px 18px 18px",
                }}
            >
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
                    <div style={{
                        width: 22, height: 22, borderRadius: 6,
                        background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                        <Sparkles size={12} color="white" />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-secondary)", letterSpacing: "0.5px" }}>LUCIO AI</span>
                </div>

                <p style={{ color: "var(--text-primary)", lineHeight: 1.75, fontSize: 14, margin: "0 0 14px" }}>
                    {item.answer}
                </p>

                {/* Citations */}
                {item.sources.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
                        {item.sources.map((s, i) => <CitationBadge key={i} {...s} />)}
                    </div>
                )}

                {/* Toggle retrieved chunks */}
                {item.chunks.length > 0 && (
                    <button
                        onClick={() => setShowChunks((v) => !v)}
                        style={{
                            background: "none", border: "1px solid var(--border)", borderRadius: 8,
                            cursor: "pointer", color: "var(--text-secondary)", fontSize: 11,
                            padding: "5px 12px", display: "flex", alignItems: "center", gap: 5,
                        }}
                    >
                        {showChunks ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        {item.chunks.length} source excerpt{item.chunks.length !== 1 ? "s" : ""}
                    </button>
                )}

                {showChunks && (
                    <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                        {item.chunks.map((c, i) => <ChunkCard key={i} chunk={c} idx={i} />)}
                    </div>
                )}
            </div>
        </div>
    );
}

function StreamingAnswer({ question, onComplete }: { question: string; onComplete: (item: HistoryItem) => void }) {
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [answerText, setAnswerText] = useState("");
    const [sources, setSources] = useState<Source[]>([]);
    const [done, setDone] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showChunks, setShowChunks] = useState(false);
    const stopRef = useRef<(() => void) | null>(null);

    useEffect(() => {
        const stop = streamQuery(question, (evt: StreamEvent) => {
            if (evt.type === "chunk") {
                setChunks((prev) => [...prev, { doc: evt.doc, page: evt.page, text: evt.text }]);
            } else if (evt.type === "answer") {
                setAnswerText(evt.text);
            } else if (evt.type === "sources") {
                setSources(evt.sources);
            } else if (evt.type === "done") {
                setDone(true);
                setAnswerText((a) => {
                    onComplete({ id: Date.now().toString(), question, answer: a, sources, chunks });
                    return a;
                });
            } else if (evt.type === "error") {
                setError(evt.message);
                setDone(true);
            }
        });
        stopRef.current = stop;
        return () => stop();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div style={{ marginBottom: 28 }}>
            {/* Question bubble */}
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
                <div style={{
                    maxWidth: "75%", padding: "10px 16px", borderRadius: "18px 18px 4px 18px",
                    background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                    color: "white", fontSize: 14, lineHeight: 1.5, fontWeight: 500,
                }}>
                    {question}
                </div>
            </div>

            {/* Answer */}
            <div style={{
                maxWidth: "85%", padding: "16px 20px",
                background: "var(--bg-card)", border: "1px solid var(--border)",
                borderRadius: "4px 18px 18px 18px",
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
                    <div style={{
                        width: 22, height: 22, borderRadius: 6,
                        background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                        <Sparkles size={12} color="white" />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-secondary)", letterSpacing: "0.5px" }}>LUCIO AI</span>
                    {!done && (
                        <span className="pulse-dot" style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", marginLeft: 4 }} />
                    )}
                </div>

                {/* Chunks streamed first */}
                {chunks.length > 0 && !answerText && (
                    <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                            Retrieving {chunks.length} source excerpt{chunks.length !== 1 ? "s" : ""}...
                        </div>
                        <div className="shimmer" style={{ height: 40, borderRadius: 8 }} />
                    </div>
                )}

                {error ? (
                    <p style={{ color: "var(--error)", fontSize: 13 }}>{error}</p>
                ) : (
                    <p className={!done ? "typing-cursor" : ""} style={{ color: "var(--text-primary)", lineHeight: 1.75, fontSize: 14, margin: "0 0 12px" }}>
                        {answerText || (!done ? "" : "Thinking...")}
                    </p>
                )}

                {sources.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
                        {sources.map((s, i) => <CitationBadge key={i} {...s} />)}
                    </div>
                )}

                {chunks.length > 0 && done && (
                    <button
                        onClick={() => setShowChunks((v) => !v)}
                        style={{
                            background: "none", border: "1px solid var(--border)", borderRadius: 8,
                            cursor: "pointer", color: "var(--text-secondary)", fontSize: 11,
                            padding: "5px 12px", display: "flex", alignItems: "center", gap: 5,
                        }}
                    >
                        {showChunks ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        {chunks.length} source excerpt{chunks.length !== 1 ? "s" : ""}
                    </button>
                )}
                {showChunks && done && (
                    <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                        {chunks.map((c, i) => <ChunkCard key={i} chunk={c} idx={i} />)}
                    </div>
                )}
            </div>
        </div>
    );
}

const SUGGESTIONS = [
    "What are the payment terms?",
    "What are the termination clauses?",
    "What governing law applies?",
    "What are the confidentiality obligations?",
    "What are the indemnification provisions?",
];

export function QueryPanel() {
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [activeQuestion, setActiveQuestion] = useState<string | null>(null);
    const [inputValue, setInputValue] = useState("");
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [history, activeQuestion]);

    const handleSubmit = () => {
        const q = inputValue.trim();
        if (!q || loading) return;
        setInputValue("");
        setLoading(true);
        setActiveQuestion(q);
    };

    const handleComplete = (item: HistoryItem) => {
        setHistory((prev) => [...prev, item]);
        setActiveQuestion(null);
        setLoading(false);
    };

    const handleKey = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
            {/* Top bar */}
            <div style={{
                padding: "16px 24px", borderBottom: "1px solid var(--border)",
                display: "flex", alignItems: "center", justifyContent: "space-between",
                background: "var(--bg-secondary)",
            }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: 16, fontWeight: 700, letterSpacing: "-0.3px" }}>Document Research</h1>
                    <p style={{ margin: 0, fontSize: 12, color: "var(--text-secondary)" }}>Ask questions · Get cited answers</p>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-muted)", background: "var(--bg-card)", padding: "5px 12px", borderRadius: 999, border: "1px solid var(--border)" }}>
                    <span className="pulse-dot" style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--success)" }} />
                    Live
                </div>
            </div>

            {/* Messages */}
            <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
                {history.length === 0 && !activeQuestion && (
                    <div style={{ textAlign: "center", paddingTop: 60 }}>
                        <div style={{
                            width: 60, height: 60, borderRadius: 18, margin: "0 auto 16px",
                            background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            boxShadow: "0 0 40px var(--accent-glow)",
                        }}>
                            <Sparkles size={28} color="white" />
                        </div>
                        <h2 style={{ margin: "0 0 8px", fontSize: 22, fontWeight: 700, letterSpacing: "-0.5px" }}>
                            Ask anything about your documents
                        </h2>
                        <p style={{ color: "var(--text-secondary)", fontSize: 14, margin: "0 0 32px" }}>
                            Every answer cites the exact source document and page.
                        </p>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                            {SUGGESTIONS.map((s) => (
                                <button
                                    key={s}
                                    onClick={() => { setInputValue(s); inputRef.current?.focus(); }}
                                    style={{
                                        padding: "8px 16px", borderRadius: 999, fontSize: 13,
                                        background: "var(--bg-card)", border: "1px solid var(--border)",
                                        color: "var(--text-secondary)", cursor: "pointer",
                                        transition: "all 0.15s",
                                    }}
                                >
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {history.map((item) => <AnswerBlock key={item.id} item={item} />)}
                {activeQuestion && (
                    <StreamingAnswer
                        key={activeQuestion + Date.now()}
                        question={activeQuestion}
                        onComplete={handleComplete}
                    />
                )}
                <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div style={{ padding: "16px 24px", borderTop: "1px solid var(--border)", background: "var(--bg-secondary)" }}>
                <div
                    className="glow-focus"
                    style={{
                        display: "flex", alignItems: "flex-end", gap: 10,
                        background: "var(--bg-card)", border: "1px solid var(--border)",
                        borderRadius: 14, padding: "10px 14px", transition: "all 0.2s",
                    }}
                >
                    <textarea
                        ref={inputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKey}
                        placeholder="Ask a legal question… (Enter to send, Shift+Enter for newline)"
                        rows={1}
                        style={{
                            flex: 1, background: "none", border: "none", outline: "none",
                            color: "var(--text-primary)", fontSize: 14, resize: "none",
                            fontFamily: "inherit", lineHeight: 1.5, maxHeight: 120, overflowY: "auto",
                        }}
                    />
                    <button
                        onClick={handleSubmit}
                        disabled={!inputValue.trim() || loading}
                        style={{
                            width: 38, height: 38, borderRadius: 10, border: "none", cursor: "pointer",
                            background: inputValue.trim() && !loading
                                ? "linear-gradient(135deg, var(--accent), #7c3aed)"
                                : "var(--bg-hover)",
                            color: "white",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            transition: "all 0.2s", flexShrink: 0,
                        }}
                    >
                        {loading
                            ? <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
                            : <Send size={16} />
                        }
                    </button>
                </div>
                <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-muted)", textAlign: "center" }}>
                    Answers are grounded in your uploaded documents only.
                </div>
            </div>
        </div>
    );
}
