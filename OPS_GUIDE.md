# Lucio Challenge — GPU Operations Guide

This one-page guide explains how to rent a cloud GPU and run the Lucio Challenge pipeline at full speed.

---

## Recommended Hardware

| GPU | VRAM | Use Case | Estimated Speedup vs CPU |
|---|---|---|---|
| **A10G** (24 GB) | 24 GB | 200 docs, full embeddings | ~15–20x |
| **RTX 3090** (24 GB) | 24 GB | Same, cheaper | ~12–18x |
| **A100** (80 GB) | 80 GB | Massive corpora, 1000+ docs | ~30x |

---

## Option 1: Lambda Labs (Recommended — easiest)

1. Go to [lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
2. Sign in and navigate to **Cloud GPUs**.
3. Select an **A10G (24 GB)** instance — typically **$0.75/hour**.
4. Choose **Ubuntu 22.04** as the OS image.
5. Add your SSH public key and launch the instance.
6. SSH in: `ssh ubuntu@<your-instance-ip>`

---

## Option 2: Vast.ai (Cheapest)

1. Go to [vast.ai](https://vast.ai) and sign up.
2. Click **Search Offers** and filter by:
   - GPU: RTX 3090 or A10G
   - CUDA >= 11.8
3. Rent the instance, download the SSH command from the dashboard.
4. SSH in using the provided command.

---

## Option 3: RunPod

1. Go to [runpod.io](https://runpod.io) and sign in.
2. Click **Deploy** and choose **GPU Pod**.
3. Select **RTX 3090** or **A10G**, and deploy with a **PyTorch** template.
4. Open the terminal from the RunPod dashboard.

---

## Setup on the Rented Machine

Once connected via SSH, run these commands:

```bash
# 1. Clone / upload your project
git clone https://your-git-repo/lucio.git   # OR scp lucio.zip ubuntu@ip:~
unzip lucio.zip && cd lucio

# 2. Install dependencies
pip install -r requirements.txt
pip install tiktoken networkx sentence-transformers chromadb google-genai openpyxl

# 3. Set your API key
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env
echo "GPU_DEVICE=cuda" >> .env

# 4. Run the full challenge pipeline
python run_challenge.py --force --strategy token --benchmark
```

> With an A10G, the ~265,000-chunk embedding build that takes 2 hours on CPU will complete in about **6–8 minutes**.

---

## Expected Runtimes on A10G (200 docs, ~265k chunks)

| Stage | CPU Time | A10G Time |
|---|---|---|
| Ingestion (6 workers) | ~2 min | ~2 min |
| Embedding generation | ~90 min | ~6–8 min |
| FAISS index build | ~30 s | ~10 s |
| Retrieval + reranking (15 q) | ~10 s | ~3 s |
| LLM answering (async, 15 q) | ~30 s | ~30 s |
| **Total** | **~2 hrs** | **~10–15 min** |

---

## Cost Estimate

| Provider | GPU | Cost/hr | Est. total run |
|---|---|---|---|
| Lambda Labs | A10G | $0.75 | $0.15–0.20 |
| Vast.ai | RTX 3090 | $0.30–0.50 | $0.08–0.15 |
| RunPod | A10G | $0.69 | $0.15–0.20 |

> **Tip:** Stop the instance immediately after `run_challenge.py` finishes to avoid extra charges.
