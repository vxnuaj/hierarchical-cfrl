"""Upload the lean checkpoint backup (~/hcfrl_backup) to a PRIVATE Hugging Face model repo.
Token: read from the HF_TOKEN environment variable only (never written to disk on the cluster).
usage (on the cluster):  read T < /dev/stdin; HF_TOKEN=$T python scripts/hf_upload.py <user>/<repo> [folder]
"""
import os, sys, time
from huggingface_hub import HfApi

repo = sys.argv[1]; folder = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/hcfrl_backup")
tok = os.environ.get("HF_TOKEN"); assert tok, "HF_TOKEN not set"
api = HfApi(token=tok)
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
readme = os.path.join(folder, "README.md")
if not os.path.exists(readme):
    open(readme, "w").write("# hcfrl checkpoints (private backup)\n\nSelected SFT bases (`sft/`), best critic epochs (`critic/`), "
                            "final + best policies of every RL run (`rl/`), analysis outputs (`analysis/`), base evals (`eval/`). "
                            "Run records, configs and the result→run map are in the code repository (`runs/`, `notes/REPRODUCE.md`).\n")
t0 = time.time()
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=folder, private=True)
print(f"UPLOAD_DONE {repo} in {time.time()-t0:.0f}s", flush=True)
