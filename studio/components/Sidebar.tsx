"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import {
    fetchStatus,
    fetchDocuments,
    triggerBuild,
    streamBuildProgress,
    StatusResult,
    DocFile,
} from "@/lib/api";
import {
    FileText, FileCode, Globe, BookOpen,
    CheckCircle, AlertCircle, Loader2, RefreshCw,
    ChevronRight, Database, Zap,
} from "lucide-react";

interface SidebarProps {
    selectedDoc: string | null;
    onDocSelect: (name: string) => void;
}

function DocIcon({ type }: { type: string }) {
    const props = { size: 14, strokeWidth: 1.8 };
    if (type === "pdf") return <FileText {...props} style={{ color: "#f87171" }} />;
    if (type === "docx" || type === "doc") return <FileCode {...props} style={{ color: "#60a5fa" }} />;
    if (type === "html" || type === "htm") return <Globe {...props} style={{ color: "#34d399" }} />;
    return <BookOpen {...props} style={{ color: "#a78bfa" }} />;
}

function StatusBadge({ ready }: { ready: boolean }) {
    return (
        <span
            style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                fontSize: 11,
                fontWeight: 600,
                padding: "2px 8px",
                borderRadius: 999,
                background: ready ? "rgba(52,211,153,0.1)" : "rgba(251,191,36,0.1)",
                color: ready ? "var(--success)" : "var(--warning)",
                border: `1px solid ${ready ? "rgba(52,211,153,0.25)" : "rgba(251,191,36,0.25)"}`,
            }}
        >
            {ready
                ? <><CheckCircle size={10} /> Ready</>
                : <><AlertCircle size={10} /> Not indexed</>
            }
        </span>
    );
}

export function Sidebar({ selectedDoc, onDocSelect }: SidebarProps) {
    const [status, setStatus] = useState<StatusResult | null>(null);
    const [docs, setDocs] = useState<DocFile[]>([]);
    const [building, setBuilding] = useState(false);
    const [buildLog, setBuildLog] = useState<string[]>([]);
    const [showLog, setShowLog] = useState(false);
    const logRef = useRef<HTMLDivElement>(null);

    const refresh = useCallback(async () => {
        try {
            const [s, d] = await Promise.all([fetchStatus(), fetchDocuments()]);
            setStatus(s);
            setDocs(d);
        } catch { /* ignore */ }
    }, []);

    useEffect(() => { refresh(); const id = setInterval(refresh, 10000); return () => clearInterval(id); }, [refresh]);

    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [buildLog]);

    const handleBuild = async (force = false) => {
        setBuilding(true);
        setBuildLog([]);
        setShowLog(true);
        try {
            await triggerBuild(force);
            const stop = streamBuildProgress(
                (line) => setBuildLog((prev) => [...prev, line]),
                () => { setBuilding(false); refresh(); stop(); },
                (err) => { setBuilding(false); setBuildLog((p) => [...p, `ERROR: ${err}`]); stop(); }
            );
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            setBuildLog((p) => [...p, `Error: ${msg}`]);
            setBuilding(false);
        }
    };

    return (
        <aside
            style={{
                width: 280,
                flexShrink: 0,
                display: "flex",
                flexDirection: "column",
                height: "100vh",
                background: "var(--bg-secondary)",
                overflow: "hidden",
            }}
        >
            {/* Header */}
            <div style={{ padding: "20px 18px 14px", borderBottom: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                    <div
                        style={{
                            width: 32, height: 32, borderRadius: 8,
                            background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                    >
                        <Zap size={16} color="white" />
                    </div>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.3px" }}>Lucio Studio</div>
                        <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>Legal Document QA</div>
                    </div>
                </div>
            </div>

            {/* Index status */}
            <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-secondary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.6px" }}>
                        <Database size={12} /> Index Status
                    </div>
                    <button
                        onClick={refresh}
                        style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-secondary)", padding: 2 }}
                    >
                        <RefreshCw size={13} />
                    </button>
                </div>

                {status ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <StatusBadge ready={status.ready} />
                            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                                    {status.doc_count} docs · {status.chunk_count.toLocaleString()} chunks
                                </span>
                                {status.build_duration_seconds != null && (
                                    <span style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                                        Built in {status.build_duration_seconds}s
                                    </span>
                                )}
                            </div>
                        </div>
                        <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                            <button
                                onClick={() => handleBuild(true)}
                                disabled={building}
                                style={{
                                    flex: 1, padding: "7px 0", borderRadius: 8, fontSize: 12, fontWeight: 600,
                                    cursor: building ? "not-allowed" : "pointer",
                                    background: "linear-gradient(135deg, var(--accent), #7c3aed)",
                                    color: "white", border: "none",
                                    opacity: building ? 0.6 : 1,
                                    display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
                                }}
                            >
                                {building ? <><Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> Building...</> : <><Zap size={12} /> Build Index</>}
                            </button>
                            <button
                                onClick={() => handleBuild(true)}
                                disabled={building}
                                title="Force full rebuild"
                                style={{
                                    padding: "7px 10px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                                    cursor: building ? "not-allowed" : "pointer",
                                    background: "var(--bg-hover)", color: "var(--text-secondary)",
                                    border: "1px solid var(--border)", opacity: building ? 0.5 : 1,
                                }}
                            >
                                <RefreshCw size={12} />
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="shimmer" style={{ height: 52, borderRadius: 8 }} />
                )}

                {/* Build log */}
                {showLog && buildLog.length > 0 && (
                    <div
                        ref={logRef}
                        style={{
                            marginTop: 10, maxHeight: 120, overflowY: "auto",
                            background: "var(--bg-primary)", borderRadius: 8,
                            padding: "8px 10px", fontSize: 10, fontFamily: "monospace",
                            color: "var(--text-secondary)", border: "1px solid var(--border)",
                        }}
                    >
                        {buildLog.map((line, i) => <div key={i}>{line}</div>)}
                    </div>
                )}
            </div>

            {/* Document list */}
            <div style={{ flex: 1, overflowY: "auto", padding: "12px 0" }}>
                <div style={{ padding: "0 18px 8px", fontSize: 10, color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.8px" }}>
                    Documents ({docs.length})
                </div>
                {docs.length === 0 ? (
                    <div style={{ padding: "20px 18px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
                        No documents found.<br />
                        <span style={{ fontSize: 11 }}>Add files to <code>documents/</code></span>
                    </div>
                ) : (
                    docs.map((doc) => (
                        <button
                            key={doc.name}
                            onClick={() => onDocSelect(doc.name)}
                            style={{
                                width: "100%", textAlign: "left",
                                padding: "8px 18px", background: selectedDoc === doc.name ? "var(--bg-hover)" : "transparent",
                                border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 8,
                                borderLeft: selectedDoc === doc.name ? "2px solid var(--accent)" : "2px solid transparent",
                                transition: "all 0.15s",
                            }}
                        >
                            <DocIcon type={doc.type} />
                            <div style={{ flex: 1, overflow: "hidden" }}>
                                <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                                    {doc.name}
                                </div>
                                <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{doc.size_kb} KB</div>
                            </div>
                            {selectedDoc === doc.name && <ChevronRight size={12} style={{ color: "var(--accent)", flexShrink: 0 }} />}
                        </button>
                    ))
                )}
            </div>

            {/* Footer */}
            <div style={{ padding: "12px 18px", borderTop: "1px solid var(--border)", fontSize: 10, color: "var(--text-muted)" }}>
                Hybrid BM25 + FAISS · GPT-4
            </div>
        </aside>
    );
}
