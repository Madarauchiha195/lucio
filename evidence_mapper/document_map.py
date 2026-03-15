"""
evidence_mapper/document_map.py
In-memory document graph using networkx.

Nodes:
  - doc   : {name, path}
  - page  : {doc, number}
  - entity: {text, type}   (e.g. party name, statute, company)

Edges:
  - cites       : doc → doc
  - references  : doc → entity
  - party       : doc → entity (for parties involved)
  - appears_on  : entity → page

Public APIs:
  add_document(name)
  add_page(doc, page_number)
  add_entity(text, entity_type)
  add_citation(from_doc, to_doc)
  add_party(doc, party_name)
  documents_citing(name)   → list of doc names
  co_parties(entity)       → docs referencing same party
  entity_pages(entity)     → list of (doc, page) tuples
  extract_citations(text, doc_name)  → auto-extract legal citations
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Legal citation patterns
_CITATION_PATTERNS = [
    # Case citation: State v. Smith, 123 F.2d 456 (1990)
    re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+\d+\s+[A-Z\.]+\s+\d+"),
    # Statute: 15 U.S.C. § 1234
    re.compile(r"\d+\s+U\.S\.C\.\s+§\s*\d+[a-z]?"),
    # Section reference: Section 12(b), § 3(c)
    re.compile(r"§\s*\d+[a-z]?(?:\(\w\))?"),
]


class DocumentGraph:
    def __init__(self):
        try:
            import networkx as nx
            self.G = nx.DiGraph()
        except ImportError:
            raise ImportError("networkx not installed. Run: pip install networkx")

    def add_document(self, name: str, path: str = ""):
        self.G.add_node(f"doc:{name}", kind="doc", name=name, path=path)

    def add_page(self, doc: str, page_number: int):
        page_id = f"page:{doc}:{page_number}"
        self.G.add_node(page_id, kind="page", doc=doc, number=page_number)
        self.G.add_edge(f"doc:{doc}", page_id, relation="has_page")

    def add_entity(self, text: str, entity_type: str = "general"):
        eid = f"entity:{text.lower()}"
        self.G.add_node(eid, kind="entity", text=text, entity_type=entity_type)

    def add_citation(self, from_doc: str, to_doc: str):
        self.G.add_edge(f"doc:{from_doc}", f"doc:{to_doc}", relation="cites")

    def add_party(self, doc: str, party_name: str):
        eid = f"entity:{party_name.lower()}"
        self.add_entity(party_name, "party")
        self.G.add_edge(f"doc:{doc}", eid, relation="party")

    def add_reference(self, doc: str, entity_text: str, page_number: int = 0):
        eid = f"entity:{entity_text.lower()}"
        pid = f"page:{doc}:{page_number}"
        self.G.add_edge(eid, pid, relation="appears_on")

    def documents_citing(self, name: str) -> List[str]:
        """Return all documents that cite *name*."""
        target = f"doc:{name}"
        return [
            n.split(":", 1)[1]
            for n, _ in self.G.in_edges(target)
            if n.startswith("doc:")
        ]

    def co_parties(self, entity: str) -> List[str]:
        """Return all documents referencing the same party/entity."""
        eid = f"entity:{entity.lower()}"
        return [
            n.split(":", 1)[1]
            for n in self.G.predecessors(eid)
            if n.startswith("doc:")
        ]

    def entity_pages(self, entity: str) -> List[Tuple[str, int]]:
        """Return list of (doc_name, page_number) where entity appears."""
        eid = f"entity:{entity.lower()}"
        results = []
        for pid in self.G.successors(eid):
            if pid.startswith("page:"):
                data = self.G.nodes[pid]
                results.append((data.get("doc", ""), data.get("number", 0)))
        return results

    def extract_citations(self, text: str, doc_name: str) -> List[str]:
        """
        Auto-detect legal citations in *text* and add them to the graph.
        Returns a list of raw citation strings found.
        """
        self.add_document(doc_name)
        found = []
        for pattern in _CITATION_PATTERNS:
            for match in pattern.finditer(text):
                citation = match.group(0).strip()
                found.append(citation)
                self.add_entity(citation, "legal_citation")
                self.add_reference(doc_name, citation)
        return found

    def summary(self) -> dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "docs":  sum(1 for n, d in self.G.nodes(data=True) if d.get("kind") == "doc"),
        }
