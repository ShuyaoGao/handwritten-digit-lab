"""把 PyTorch 模型导出成前端可用的格式：float16 权重（base64）+ 层列表 + 一致性校验样本。

层列表与 common/nn.js 的 run() 一一对应；校验样本供 tests/ 用 V8 跑 nn.js 比对。
"""
import base64

import numpy as np


def pack_weights(named_state_dicts):
    """named_state_dicts: {前缀: state_dict}。返回 (manifest, base64)。"""
    manifest, chunks, off = {}, [], 0
    for prefix, sd in named_state_dicts.items():
        for k, t in sd.items():
            a = t.detach().cpu().numpy().astype(np.float16).ravel()
            manifest[f"{prefix}.{k}"] = [off, list(t.shape)]
            chunks.append(a)
            off += a.size
    blob = np.concatenate(chunks).astype("<f2").tobytes()
    return manifest, base64.b64encode(blob).decode()


def u8_b64(arr):
    """[0,1] 浮点图像数组 → uint8 → base64"""
    a = np.clip(np.round(np.asarray(arr) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(a.tobytes()).decode()


def js_assign(var, obj):
    """生成 `window.VAR = {...};`，供 <script src> 加载（file:// 下不能 fetch）"""
    import json
    return f"window.{var} = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n"
