#!/usr/bin/env bash
# start_mlflow_ui.sh -- serve the project's MLflow tracking store on a compute (interactive)
# node and print the SSH local-forward spec to use from your laptop. (Interactive companion to
# mlflow_server.lsf, which does the same under LSF with an inactivity auto-shutdown.)
#
# Why this shape: the login node forbids long-running servers, so MLflow runs on a compute
# node. We bind MLflow to 0.0.0.0 and let the login node reach it over the cluster LAN, so the
# laptop needs only ONE ordinary SSH local-forward to the login node.
#
# Usage (run from the repo dir, inside the conda env, on an interactive/compute node):
#     ./start_mlflow_ui.sh [PORT]            # PORT defaults to 5050; auto-bumps if busy
#
# Nothing HPC-specific is hardcoded: the repo dir, store URI, node name/IP and username are
# all derived at runtime. Set MLFLOW_LOGIN_HOST to have the full ssh command printed for you.
#
# NOTE (security): --host 0.0.0.0 exposes the UI/REST API to the cluster LAN while it runs.
# MLflow's Host-header allowlist is NOT real protection. Keep the port obscure; stop it when
# you're done.
set -euo pipefail

PORT="${1:-${PORT:-5050}}"

# Repo dir = this script's own directory, so the store path resolves regardless of CWD.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Serve exactly what the pipeline writes to (sqlite mlflow.db on mlflow>=3.13, or the ./mlruns
# file store on older pins) -- ask the env so viewer and writer can never disagree.
STORE="${MLFLOW_BACKEND_URI:-$(python -c 'import mlflow; print(mlflow.get_tracking_uri())')}"

# First free port at/after $PORT (the node is shared).
while ss -ltn 2>/dev/null | grep -q ":${PORT} "; do
  echo "port ${PORT} in use, trying $((PORT + 1))" >&2
  PORT=$((PORT + 1))
done

# Connection target, derived at runtime (FQDN, with IP as a fallback if the name doesn't
# resolve from the login node).
NODE_FQDN="$(hostname -f 2>/dev/null || hostname)"
NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

cat >&2 <<EOF

============================================================
 MLflow UI starting (store: ${STORE})

 On your LAPTOP, add this forward to your login SSH:

     ${PORT}:${NODE_FQDN}:${PORT}
   (or, if that name won't resolve from the login node, use the IP:)
     ${PORT}:${NODE_IP}:${PORT}

     ssh -N -L ${PORT}:${NODE_FQDN}:${PORT} $(whoami)@${MLFLOW_LOGIN_HOST:-<your-login-host>}

 then open  http://127.0.0.1:${PORT}
 (Ctrl-C here stops the server.)
============================================================

EOF

exec mlflow ui --backend-store-uri "$STORE" --host 0.0.0.0 --port "${PORT}"
