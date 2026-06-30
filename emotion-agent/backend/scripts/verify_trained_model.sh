#!/usr/bin/env bash
# 快速判断后端是否绑定训练 checkpoint（非 mock）
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"

echo "=== GET /api/v1/model/status ==="
curl -s "${BASE}/api/v1/model/status" | python3 -m json.tool

echo ""
echo "=== 判定说明 ==="
python3 -c "
import json, urllib.request
base = '${BASE}'
with urllib.request.urlopen(base + '/api/v1/model/status') as r:
    d = json.load(r)
ok = d.get('using_trained_checkpoint')
print('using_trained_checkpoint =', ok)
if ok:
    m = d.get('model', {})
    print('结论: 已加载训练模型，可进行真实推理')
    print('  preset=', m.get('preset'))
    print('  checkpoint=', m.get('checkpoint_path'))
else:
    print('结论: 未使用训练 checkpoint')
    print('  MODEL_PROVIDER(env)=', d.get('model_provider_env'))
    print('  loaded=', d.get('model', {}).get('loaded'))
"
