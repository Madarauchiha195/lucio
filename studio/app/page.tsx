"use client";
import { Sidebar } from "@/components/Sidebar";
import { QueryPanel } from "@/components/QueryPanel";
import { useState } from "react";

export default function HomePage() {
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
        background: "var(--bg-primary)",
      }}
    >
      {/* Sidebar */}
      <Sidebar onDocSelect={setSelectedDoc} selectedDoc={selectedDoc} />

      {/* Divider */}
      <div
        style={{
          width: "1px",
          background: "var(--border)",
          flexShrink: 0,
        }}
      />

      {/* Main panel */}
      <main style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <QueryPanel />
      </main>
    </div>
  );
}
