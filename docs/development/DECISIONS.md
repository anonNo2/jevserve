# Decision log

| ID | Decision | Reason |
|---|---|---|
| D001 | Brand JevServe; technical names jevserve | User selected the name and accepted similarity to JevServe-Bench |
| D002 | Linux NVIDIA, single host, Docker/Compose | Explicit user deployment constraint |
| D003 | Single image/container by default | Meet the vLLM-like minimal startup experience |
| D004 | vLLM first; bounded adapters | Reuse the inference engine and focus on correct model integration |
| D005 | No Triton implementation or Kubernetes in alpha | Avoid expanding scope before one real model works |
| D006 | Real GPU evidence required | Bootstrap or CPU-only success does not satisfy the product goal |
| D007 | Private host notes in ignored .local/ | Preserve development context without committing private infrastructure details |
| D008 | Planning docs are historical inputs to T00 | Upstream versions, models and docs need direct verification before use |

Add material changes with evidence, compatibility consequences and affected tasks. Do not treat this log as permission to change the user's agreed scope silently.
