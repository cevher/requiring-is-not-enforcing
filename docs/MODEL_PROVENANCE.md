# Model provenance

Recording which weights produced which generations. The main set was run under a
moving tag and its digest was not captured; this file states that plainly and
pins everything produced afterwards.

## Primary model

| | Main generation set | Required-role condition and `full` rerun |
|---|---|---|
| Tag requested | `llama3:latest` | `llama3:8b-instruct-q4_0` |
| Tag type | moving | explicit, non-moving |
| Digest recorded at run time | **no** | yes (below) |
| Run date | 2026-04 | 2026-10 |
| Provider entry | `ollama_local_llama` | `ollama_llama3_q4_pinned` |
| Adapter | `ollama_native` | `ollama_native` |
| Decoding | `format: json`, temp 0.1, 1200-token limit | identical |

Verified 2026-10-26 with `ollama show llama3:8b-instruct-q4_0`:

```
architecture       llama
parameters         8.0B
context length     8192
embedding length   4096
quantization       Q4_0
```

Blob digests from `ollama pull`: model layer `sha256:6a0746a1ec1a...` (4.7 GB),
parameter layer `sha256:3f8eb4da87fa...` (485 B).
Upstream: Meta Llama 3, release date 2024-04-18.

## Why the `full` condition was rerun

The original Ollama installation was removed before the digest of `llama3:latest`
could be read, so it cannot be shown that the April and October runs used the same
weights. Comparing a `full` condition generated in April against a
`full_required_role` condition generated in October would confound the manipulation
with a possible change of model version.

The `full` condition was therefore regenerated under the pinned tag, so that both
arms of the comparison come from weights that are identical and recorded. The
April set is retained unchanged and reported separately.

## Secondary model

`openai/gpt-oss-20b` (MXFP4 GGUF) served through LM Studio 0.4.25 under an open
JSON schema, temperature 0.1, 4,096-token limit. Provider entry
`lmstudio_gptoss20b_jsonmode`. Not reinstalled, so the weights are unchanged from
the main run.

## Pilot runs not reported in the paper

The repository also contains pilot generations from `google/gemma-4-e4b` (24 outputs)
and `qwen/qwen3-coder-30b` (4 outputs). These were prompt- and schema-debugging runs
made while the compact output schema was being settled. They are kept so the record
is complete and are used for no reported result.
