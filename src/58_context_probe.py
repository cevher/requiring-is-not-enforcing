"""
58_context_probe.py  (teshis modu)

Does the serving runtime see the whole prompt?

The generation runs never set num_ctx, so the Ollama server used its own default
context window. The full and full_required_role prompts are ~3,700-3,800 tokens.
If the default was below that, the runtime truncated the prompt from the front,
where the system instruction sits.

prompt_eval_count in the raw API response is the number of prompt tokens the
model actually evaluated. 28_run_llm_generation.py discards it, so it cannot be
recovered from past runs. This probe sends one real prompt under several payload
variants and reports prompt_eval_count for each, with the server's error body
printed in full when a variant fails.

    python src/58_context_probe.py
"""

import json, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
CFG  = json.load(open(ROOT / "llm_provider_config.json", encoding="utf-8"))
PROMPTS = ROOT / "data_processed" / "evidence" / "llm_prompt_conditions_required_role.jsonl"

hits = [p for p in CFG["providers"] if p["provider_name"] == "ollama_llama3_q8_pinned"]
if not hits:
    sys.exit("saglayici yok: ollama_llama3_q8_pinned")
p = hits[0]
base = p["base_url"].rstrip("/")
root = base[:-4] if base.endswith("/api") else base

rec  = json.loads(open(PROMPTS, encoding="utf-8").readline())
msgs = rec["messages"]
chars = sum(len(m["content"]) for m in msgs)

try:
    ver = requests.get(root + "/api/version", timeout=10).json().get("version")
except Exception as e:
    ver = f"okunamadi ({e})"

print(f"Ollama surumu : {ver}")
print(f"Model         : {p['model']}")
print(f"Prompt        : {rec['prompt_id']}")
print(f"Prompt boyutu : {chars} karakter (~{chars/3.6:.0f} token)")
print("-" * 78)

T = p.get("temperature", 0.1)
N = p.get("max_tokens", 1200)

VARIANTS = [
    ("A  28'in yaptigi aynen (format=json)",
     {"model": p["model"], "messages": msgs, "stream": False,
      "options": {"temperature": T, "num_predict": N}, "format": "json"}),
    ("B  A + num_ctx=8192",
     {"model": p["model"], "messages": msgs, "stream": False,
      "options": {"temperature": T, "num_predict": N, "num_ctx": 8192}, "format": "json"}),
    ("C  format yok, options var",
     {"model": p["model"], "messages": msgs, "stream": False,
      "options": {"temperature": T, "num_predict": N}}),
    ("D  ciplak (sadece model+messages)",
     {"model": p["model"], "messages": msgs, "stream": False}),
]

res = {}
for label, payload in VARIANTS:
    try:
        r = requests.post(base + "/chat", json=payload, timeout=p.get("timeout", 240))
    except Exception as e:
        print(f"{label:38s} BAGLANTI HATASI: {e!r}")
        continue
    if not r.ok:
        print(f"{label:38s} HTTP {r.status_code}")
        print(f"{'':38s} govde: {r.text[:500]}")
        continue
    d = r.json()
    pec = d.get("prompt_eval_count")
    res[label[0]] = pec
    print(f"{label:38s} HTTP 200  prompt_eval_count = {pec}  eval_count = {d.get('eval_count')}")

print("-" * 78)
a, b = res.get("A"), res.get("B")
if a is None or b is None:
    c = res.get("C")
    if c is not None and b is not None:
        a = c
        print("(A basarisiz; C varyanti A'nin yerine kullaniliyor)")
if a is None or b is None:
    print("SONUC: karsilastirma yapilamadi. Yukaridaki govdeyi gonder.")
elif a < b:
    print(f"SONUC: KIRPILMA VAR. {b-a} prompt tokeni modele ulasmamis.")
    print("       Ollama onden kirpar; sistem talimati promptun basinda.")
else:
    print("SONUC: kirpilma yok. Sunucu varsayilani prompt'un tamamini aliyor.")
