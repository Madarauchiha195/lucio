// lib/api.ts
// API client for the Lucio Studio backend.

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface DocFile {
    name: string;
    size_kb: number;
    type: string;
}

export interface StatusResult {
    ready: boolean;
    bm25_ready: boolean;
    faiss_ready: boolean;
    doc_count: number;
    chunk_count: number;
    build_duration_seconds: number | null;
}

export interface Source {
    doc: string;
    page: number;
}

export interface Chunk {
    doc: string;
    page: number;
    text: string;
}

export type StreamEvent =
    | { type: "chunk"; doc: string; page: number; text: string }
    | { type: "answer"; text: string }
    | { type: "sources"; sources: Source[] }
    | { type: "done" }
    | { type: "error"; message: string };

// ── REST calls ────────────────────────────────────────────────────────────────

export async function fetchStatus(): Promise<StatusResult> {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) throw new Error("Failed to fetch status");
    return r.json();
}

export async function fetchDocuments(): Promise<DocFile[]> {
    const r = await fetch(`${API}/api/documents`);
    if (!r.ok) throw new Error("Failed to fetch documents");
    const data = await r.json();
    return data.documents;
}

export async function triggerBuild(force = false): Promise<void> {
    const r = await fetch(`${API}/api/build?force=${force}`, { method: "POST" });
    if (!r.ok && r.status !== 409) throw new Error("Failed to trigger build");
}

// ── SSE streams ───────────────────────────────────────────────────────────────

export function streamBuildProgress(
    onLine: (line: string) => void,
    onDone: () => void,
    onError: (e: string) => void
): () => void {
    const es = new EventSource(`${API}/api/build/progress`);
    es.onmessage = (e) => {
        if (e.data === "__DONE__") { es.close(); onDone(); }
        else if (e.data.startsWith("__ERROR__")) { es.close(); onError(e.data.slice(9)); }
        else if (e.data === "__PING__") { /* keep-alive */ }
        else onLine(e.data);
    };
    es.onerror = () => { es.close(); onError("Connection lost"); };
    return () => es.close();
}

export function streamQuery(
    question: string,
    onEvent: (e: StreamEvent) => void
): () => void {
    // POST first, then read SSE (fetch + ReadableStream for POST body)
    let aborted = false;
    const controller = new AbortController();

    (async () => {
        try {
            const res = await fetch(`${API}/api/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
                signal: controller.signal,
            });

            const reader = res.body!.getReader();
            const dec = new TextDecoder();
            let buf = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done || aborted) break;
                buf += dec.decode(value, { stream: true });
                const parts = buf.split("\n\n");
                buf = parts.pop()!;
                for (const part of parts) {
                    const line = part.replace(/^data: /, "").trim();
                    if (!line) continue;
                    try {
                        const evt: StreamEvent = JSON.parse(line);
                        onEvent(evt);
                    } catch { /* malformed */ }
                }
            }
        } catch (err: unknown) {
            if (!aborted) {
                const msg = err instanceof Error ? err.message : String(err);
                onEvent({ type: "error", message: msg });
            }
        }
    })();

    return () => { aborted = true; controller.abort(); };
}
