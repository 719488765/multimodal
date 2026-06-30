#!/usr/bin/env python3
"""Generate spacious system architecture figure + export PNG/JPG."""
from pathlib import Path

DIR = Path(__file__).parent
OUT_SVG = DIR / "system_architecture_figure.svg"
OUT_PNG = DIR / "system_architecture_figure.png"
OUT_JPG = DIR / "system_architecture_figure.jpg"

VW = 2000
GAP, A_PAD, B_PAD, C_H = 18, 12, 14, 160
ARROW_INSET = 6
CHAIN_GAP = 14
LAYER_H = 22
FONT = "Noto Sans CJK SC, Noto Serif CJK SC, WenQuanYi Micro Hei, sans-serif"
FONT_EN = "DejaVu Sans, Liberation Sans, Arial, sans-serif"

boxes, arrows, labels, fills_bg = [], [], [], []


def T(s, x, y, anchor="start", fs=9, bold=False, fill="#222", family=FONT):
    fw = "700" if bold else ("600" if fs <= 9 else "400")
    ff = f' font-family="{family}"'
    labels.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
        f'font-size="{fs}" font-weight="{fw}" fill="{fill}"{ff}>{s}</text>'
    )


def bg(x, y, w, h, fill="#f7f7f7", stroke="#bbb", sw=1.2, rx=4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    fills_bg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def B(x, y, w, h, fill="#fff", stroke="#555", sw=1, rx=3, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    boxes.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )
    return {"x": x, "y": y, "w": w, "h": h}


def cx(b):
    return b["x"] + b["w"] / 2


def cy(b):
    return b["y"] + b["h"] / 2


def right(b):
    return b["x"] + b["w"]


def bottom(b):
    return b["y"] + b["h"]


def layer(x, y, w, text, fill="#b8d4ef", stroke="#3d6ea8", h=LAYER_H, english=False):
    b = B(x, y, w, h, fill, stroke, 1, 2)
    fam = FONT_EN if english or text.isascii() else FONT
    T(text, cx(b), cy(b), anchor="middle", fs=8, bold=True, family=fam)
    return b


def ah(x1, y, x2, color="#333", marker="mk", dash=""):
    if x2 - x1 < 8:
        return
    d = f' stroke-dasharray="{dash}"' if dash else ""
    arrows.append(
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - ARROW_INSET:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="1.5" fill="none" marker-end="url(#{marker})"{d}/>'
    )


def av(x, y1, y2, color="#333", marker="mk", dash=""):
    if abs(y2 - y1) < 8:
        return
    d = f' stroke-dasharray="{dash}"' if dash else ""
    ye = y2 - ARROW_INSET if y2 > y1 else y2 + ARROW_INSET
    arrows.append(
        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{ye:.1f}" '
        f'stroke="{color}" stroke-width="1.5" fill="none" marker-end="url(#{marker})"{d}/>'
    )


def arl(b1, b2, color="#333", marker="mk", dash=""):
    ah(right(b1), cy(b1), b2["x"], color, marker, dash)


def abt(b1, b2, color="#333", marker="mk", dash=""):
    av(cx(b1), bottom(b1), b2["y"], color, marker, dash)


def apath(d, color="#333", marker="mk", dash="", sw=1.5):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    arrows.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#{marker})"{ds}/>'
    )


def chain(x, y, names, fills=None, w=46, gap=CHAIN_GAP):
    fills = fills or ["#b8d4ef"] * len(names)
    bs = []
    x0 = x
    for i, (n, f) in enumerate(zip(names, fills)):
        b = layer(x0, y, w, n, f)
        if bs:
            arl(bs[-1], b)
        bs.append(b)
        x0 += w + gap
    return bs, bs[-1]


def chip(x, y, w, text, fill="#fff", stroke="#555", english=False):
    b = B(x, y, w, 20, fill, stroke, 1, 10)
    fam = FONT_EN if english or text.isascii() else FONT
    T(text, cx(b), cy(b), anchor="middle", fs=8.5, bold=True, family=fam)
    return b


def encoder_row(ex, ry, ico, backbone, proj, out_sym, note, bb_w=62, mid_w=64, out_w=48):
    icon(ico, ex + 8, ry + 2)
    bb = layer(ex + 26, ry, bb_w, backbone, "#9ec5e8", "#3d6ea8", english=True)
    mid = layer(ex + 26 + bb_w + 6, ry, mid_w, proj, "#b8d4ef", "#3d6ea8")
    out = layer(ex + 26 + bb_w + 6 + mid_w + 6, ry, out_w, out_sym, "#c8e6c9", "#388e3c", english=True)
    arl(bb, mid)
    arl(mid, out)
    T(note, ex + 10, ry + 28, fs=7.5, fill="#555")
    return out


def ap_card(ax, apy, apw, aph, title, rows, highlight=False):
    fill, stroke, sw = ("#e8f5e9", "#2e7d32", 1.8) if highlight else ("#fff", "#555", 1.2)
    ab = B(ax, apy, apw, aph, fill, stroke, sw, 3)
    T(title, cx(ab), apy + 15, anchor="middle", fs=10, bold=True)
    for i, (label, text) in enumerate(rows):
        y = apy + 33 + i * 17
        lbl_fill = "#1b5e20" if highlight and label in ("结果", "发现") else "#555"
        T(label, ax + 8, y, fs=8, bold=True, fill=lbl_fill)
        T(text, ax + 40, y, fs=8, fill="#333")
    return ab


def icon(href, x, y, s=16):
    boxes.append(f'<use href="#{href}" x="{x}" y="{y}" width="{s}" height="{s}"/>')


def softmax_bar(x, y, w=80, h=16):
    b = B(x, y, w, h, "#e8f5e9", "#388e3c", 1, 2)
    hs = [12, 20, 8, 16, 10, 22, 6]
    bw = (w - 10) / len(hs)
    for i, bh in enumerate(hs):
        boxes.append(
            f'<rect x="{x + 5 + i * bw:.1f}" y="{y + h - 3 - bh * 0.35:.1f}" '
            f'width="{bw - 1.2:.1f}" height="{bh * 0.35:.1f}" fill="#66bb6a"/>'
        )
    T("Softmax", cx(b), y + h + 10, anchor="middle", fs=7.5, bold=True, fill="#2e7d32", family=FONT_EN)
    return b


def mha_block(x, y, w=140, h=62):
    outer = B(x, y, w, h, "#ede7f6", "#5e35b1", 1.2, 3)
    T("Multi-Head Attn", cx(outer), y + 14, anchor="middle", fs=8.5, bold=True, fill="#4527a0", family=FONT_EN)
    for ox, t in [(14, "Q"), (54, "K"), (94, "V")]:
        layer(x + ox, y + 26, 32, t, "#d1c4e9", "#5e35b1", 20, english=True)
    T("8 heads", cx(outer), y + h - 10, anchor="middle", fs=7, fill="#555", family=FONT_EN)
    return outer


DEFS = """
<marker id="mk" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8z" fill="#333"/></marker>
<marker id="mkg" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8z" fill="#2e7d32"/></marker>
<marker id="mkb" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8z" fill="#2a5080"/></marker>
<marker id="mkp" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8z" fill="#5e35b1"/></marker>
<symbol id="ico-pytorch" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#EE4C2C"/><path d="M12 5 L16 14 L8 14 Z" fill="#fff"/></symbol>
<symbol id="ico-react" viewBox="0 0 24 24"><circle cx="12" cy="12" r="2.5" fill="#61DAFB"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="#61DAFB" stroke-width="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="#61DAFB" stroke-width="1.5" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="#61DAFB" stroke-width="1.5" transform="rotate(120 12 12)"/></symbol>
<symbol id="ico-fastapi" viewBox="0 0 24 24"><rect width="24" height="24" rx="4" fill="#009688"/><path d="M6 16 L12 6 L14 12 L18 10 Z" fill="#fff"/></symbol>
<symbol id="ico-whisper" viewBox="0 0 24 24"><rect width="24" height="24" rx="4" fill="#10A37F"/><rect x="9" y="4" width="6" height="11" rx="3" fill="#fff"/><path d="M7 14 Q12 20 17 14" fill="none" stroke="#fff" stroke-width="2"/></symbol>
<symbol id="ico-ollama" viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#222"/><text x="12" y="16" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">O</text></symbol>
<symbol id="ico-hf" viewBox="0 0 24 24"><rect width="24" height="24" rx="4" fill="#FFD21E"/><circle cx="9" cy="10" r="1.5" fill="#333"/><circle cx="15" cy="10" r="1.5" fill="#333"/><path d="M8 15 Q12 18 16 15" fill="none" stroke="#333" stroke-width="1.2"/></symbol>
<symbol id="ico-cuda" viewBox="0 0 24 24"><rect width="24" height="24" rx="3" fill="#76B900"/><text x="12" y="15" text-anchor="middle" fill="#fff" font-size="7" font-weight="bold">GPU</text></symbol>
<symbol id="ico-nginx" viewBox="0 0 24 24"><rect width="24" height="24" rx="3" fill="#009639"/><text x="12" y="16" text-anchor="middle" fill="#fff" font-size="10" font-weight="bold">N</text></symbol>
<symbol id="ico-opencv" viewBox="0 0 24 24"><rect width="24" height="24" rx="3" fill="#5C3EE8"/><text x="12" y="16" text-anchor="middle" fill="#fff" font-size="8" font-weight="bold">CV</text></symbol>
<symbol id="ico-resnet" viewBox="0 0 24 24"><rect x="3" y="14" width="18" height="4" fill="#4a90d9"/><rect x="5" y="10" width="14" height="3" fill="#6aa8e8"/><rect x="7" y="6" width="10" height="3" fill="#8cc0f5"/></symbol>
<symbol id="ico-wave" viewBox="0 0 24 24"><path d="M2 12 Q6 6 10 12 T18 12 T22 12" fill="none" stroke="#5c6bc0" stroke-width="2"/></symbol>
<symbol id="ico-bert" viewBox="0 0 24 24"><rect width="24" height="24" rx="3" fill="#4285F4"/><text x="12" y="16" text-anchor="middle" fill="#fff" font-size="9" font-weight="bold">B</text></symbol>
<symbol id="ico-vite" viewBox="0 0 24 24"><path d="M12 2 L20 20 L12 16 L4 20 Z" fill="#646CFF"/></symbol>
"""


def build_svg(vh):
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VW} {vh}" '
        f'font-family="{FONT}">\n'
        f"<defs>{DEFS}</defs>\n"
        f'<rect width="{VW}" height="{vh}" fill="#ffffff"/>\n'
        f'<g id="panel-bg">{"".join(fills_bg)}</g>\n'
        f'<g id="boxes">{"".join(boxes)}</g>\n'
        f'<g id="labels">{"".join(labels)}</g>\n'
        f'<g id="arrows" fill="none">{"".join(arrows)}</g>\n'
        f"</svg>"
    )


def export_raster(svg_path, png_path, jpg_path, scale=2.0):
    import cairosvg
    from PIL import Image
    import io

    png_bytes = cairosvg.svg2png(url=str(svg_path), scale=scale)
    png_path.write_bytes(png_bytes)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.save(jpg_path, "JPEG", quality=92, optimize=True)
    return png_path, jpg_path


# ═══════════════════ (a) OFFLINE ═══════════════════
T("图1  多模态情感分析智能体系统架构", VW / 2, 28, anchor="middle", fs=18, bold=True)

AY = 46
T("(a)", 24, AY + 18, fs=13, bold=True)
T("离线训练：数据治理 → 特征编码器 → 多模态情感模型 → AP0–AP4 → 模型权重", 52, AY + 18, fs=11, bold=True)

# --- column 1: data ---
DY = AY + 34
ROW_H = 184
db = B(24, DY, 172, ROW_H, "#fafafa", "#555", 1.2, 3, "5,3")
T("数据输入", cx(db), DY + 18, anchor="middle", fs=11, bold=True)
for i, t in enumerate(["CREMA", "MELD", "MOSEI"]):
    chip(32 + i * 54, DY + 30, 50, t)
T("下载 · 整理 · 质检", 32, DY + 58, fs=9, bold=True)
T("训练 / 验证 / 测试划分", 32, DY + 74, fs=9, bold=True)
icon("ico-opencv", 32, DY + 88)
T("OpenCV 视频抽帧", 52, DY + 96, fs=9, bold=True)
icon("ico-hf", 32, DY + 108)
T("librosa 音频", 52, DY + 116, fs=9, bold=True)
icon("ico-bert", 110, DY + 108)
T("BERT 文本", 128, DY + 116, fs=9, bold=True)
B(32, DY + 126, 156, 24, "#fff8e1", "#ccc", 1, 2, "3,2")
T("7类标签 · 平衡采样", cx(db), DY + 138, anchor="middle", fs=9, bold=True)

# --- column 2: encoders — per-modality projection with explicit ops & output symbols ---
EX = 210
eb = B(EX, DY, 252, ROW_H, "#dce8f5", "#3d6ea8", 1.5, 3)
T("特征编码器", cx(eb), DY + 16, anchor="middle", fs=11, bold=True, fill="#1a4070")
T("预训练骨干 → 投影层 → 统一512维隐空间", cx(eb), DY + 30, anchor="middle", fs=8, fill="#444")
for ri, row in enumerate([
    ("ico-resnet", "ResNet-50", "池化+线性", "h_v", "4帧224² · 2048→512"),
    ("ico-wave", "Wav2Vec2", "线性+时均", "h_a", "3s波形 · 768→512"),
    ("ico-bert", "BERT", "线性+ReLU", "h_t", "token · [CLS]768→512"),
]):
    encoder_row(EX, DY + 40 + ri * 46, *row)
arl(db, eb, marker="mkb")

# --- column 3: multimodal model (align top with taller encoder) ---
MX, MY = 472, DY
MW, MH = 1016, 400
mb = B(MX, MY, MW, MH, "#fff", "#2e7d32", 2, 4)
T("多模态情感模型 · 四种可切换融合子模型", cx(mb), MY + 18, anchor="middle", fs=12, bold=True, fill="#2e7d32")
arl(eb, mb, color="#2a5080", marker="mkb")

for i, (t, col) in enumerate([("h_v", "#4a90d9"), ("h_a", "#5c6bc0"), ("h_p", "#888"), ("h_t", "#4285F4")]):
    chip(MX + 20 + i * 58, MY + 32, 48, t, "#fff", col)
T("↑ 编码器输出 · 512维模态向量", MX + 268, MY + 42, fs=9, bold=True, fill="#555")

# ① Emotion Shift — spacious 2-row internal layout
ESX, ESY = MX + 14, MY + 58
ESW, ESH = 640, 210
es = B(ESX, ESY, ESW, ESH, "#e8f5e9", "#2e7d32", 1.8, 3)
T("① Emotion Shift ★ CFN-ESA（AP2 主实验）", cx(es), ESY + 16, anchor="middle", fs=10.5, bold=True)

alpha = layer(ESX + 12, ESY + 30, 52, "alpha", "#fff9c4", "#f9a825", english=True)
modl = [layer(ESX + 72 + i * 40, ESY + 30, 34, t, "#e3f2fd", "#1976d2", english=True) for i, t in enumerate("VAPT")]
arl(alpha, modl[0])
for i in range(3):
    arl(modl[i], modl[i + 1])
hw = layer(ESX + 240, ESY + 30, 44, "h_w", "#b8d4ef", "#3d6ea8")
arl(modl[-1], hw)

esa = B(ESX + 12, ESY + 62, 328, 100, "#fff", "#666", 1, 2, "4,2")
T("情感转变感知模块", cx(esa), ESY + 74, anchor="middle", fs=9, bold=True)
_, r1 = chain(ESX + 18, ESY + 84, ["Linear", "ReLU", "Linear"], w=42, gap=10)
cls = layer(r1["x"] + r1["w"] + 10, ESY + 84, 42, "7-way", "#e8f5e9", "#388e3c", english=True)
arl(r1, cls)
softmax_bar(cls["x"] + cls["w"] + 10, ESY + 86, 50, 16)
_, r2 = chain(ESX + 18, ESY + 122, ["BiLSTM", "BiLSTM"], ["#ce93d8", "#ce93d8"], w=46, gap=10)
cc = layer(r2["x"] + r2["w"] + 10, ESY + 122, 42, "concat", "#fff9c4", "#f9a825", english=True)
arl(r2, cc)
lin = layer(cc["x"] + cc["w"] + 10, ESY + 122, 42, "Linear", "#b8d4ef", "#3d6ea8", english=True)
arl(cc, lin)
abt(hw, esa, color="#3d6ea8", dash="4,3")

mha = mha_block(ESX + 352, ESY + 88, 132, 62)
ln = layer(ESX + 492, ESY + 110, 52, "LN+res", "#c8e6c9", "#388e3c", english=True)
apath(
    f"M{right(lin)} {cy(lin)} L{right(lin) + 10} {cy(lin)} "
    f"L{right(lin) + 10} {cy(mha)} L{mha['x']} {cy(mha)}",
    color="#5e35b1",
    marker="mkp",
)
arl(mha, ln)
hf = layer(ESX + 554, ESY + 110, 52, "h_fused", "#b8d4ef", "#3d6ea8", english=True)
arl(ln, hf)
T("Q=text, K/V=video+audio", ESX + 410, ESY + 74, anchor="middle", fs=8, fill="#5e35b1", family=FONT_EN)
T("三混合 F1≈0.562", cx(es), ESY + ESH - 14, anchor="middle", fs=8.5, fill="#444")

# ② Leader-Follower — two clean rows
LFX, LFY = MX + 670, MY + 58
lf = B(LFX, LFY, 350, 100, "#fff", "#666", 1, 3)
T("② Leader-Follower · ICCV'21", cx(lf), LFY + 16, anchor="middle", fs=10, bold=True)
_, lf1 = chain(LFX + 12, LFY + 32, ["Leader Q", "Foll. K", "Foll. V"], ["#e1bee7", "#d1c4e9", "#d1c4e9"], w=48)
att = layer(lf1["x"] + lf1["w"] + 10, LFY + 32, 40, "Attn", "#ede7f6", "#5e35b1")
arl(lf1, att)
_, lf2 = chain(LFX + 12, LFY + 64, ["LF-bidir", "concat", "Fusion"], ["#e1bee7", "#fff9c4", "#b8d4ef"], w=48, gap=12)
T("AP3 融合消融", cx(lf), LFY + 88, anchor="middle", fs=8, fill="#444")

# ③ Two-Stage
TSX, TSY = MX + 670, MY + 168
ts = B(TSX, TSY, 350, 100, "#fff", "#666", 1, 3)
T("③ Two-Stage · GA2MIF", cx(ts), TSY + 16, anchor="middle", fs=10, bold=True)
t1b, t1 = chain(TSX + 12, TSY + 32, ["4-node", "GATx2", "MHA"], w=50, gap=12)
t2b, _ = chain(TSX + 12, TSY + 64, ["CrossAttn", "FFNx2", "Final"], w=50, gap=12)
av(cx(t1), bottom(t1), t2b[0]["y"], color="#888")

# ④ Standard — full width bottom strip inside model box
STX, STY = MX + 14, MY + 280
st = B(STX, STY, 640, 72, "#fff", "#666", 1, 3)
T("④ Standard 基线（AP0 对照 · F1≈0.52）", cx(st), STY + 16, anchor="middle", fs=10, bold=True)
_, s1 = chain(STX + 12, STY + 34, ["concat", "SelfAttn", "CrossAttn"], w=50)
s2b, _ = chain(s1["x"] + s1["w"] + 14, STY + 34, ["FFN+LN", "TempAttn", "pool"], w=46)
arl(s1, s2b[0])

# --- column 4 & 5: output head + output space (no cross-box overflow) ---
OX = 1504
OB_W = 162
ob = B(OX, DY, OB_W, ROW_H, "#fff8e6", "#c8a020", 1.3, 3)
T("共享输出头", cx(ob), DY + 16, anchor="middle", fs=11, bold=True)
_, oh1 = chain(OX + 10, DY + 30, ["Linear", "ReLU", "Drop"], w=36, gap=8)
T("7-way Softmax", cx(ob), DY + 58, anchor="middle", fs=8, bold=True, fill="#2e7d32", family=FONT_EN)
T("情绪分类", cx(ob), DY + 70, anchor="middle", fs=9, bold=True)
_, oh2 = chain(OX + 10, DY + 82, ["Linear", "ReLU", "2d"], w=36, gap=8)
T("效价 / 唤醒度", cx(ob), DY + 110, anchor="middle", fs=9, bold=True)
icon("ico-pytorch", OX + 10, DY + 122)
T("GPU 训练 · 选优", OX + 30, DY + 130, fs=8.5, bold=True)

apath(
    f"M{right(hf)} {cy(hf)} L{OX - 8} {cy(hf)} L{OX - 8} {DY + 46} L{OX} {DY + 46}",
    color="#2e7d32",
    marker="mkg",
)

SX = OX + OB_W + 14
SB_W = 162
sb = B(SX, DY, SB_W, ROW_H, "#edf7ed", "#388e3c", 1, 3)
T("输出空间", cx(sb), DY + 16, anchor="middle", fs=11, bold=True)
T("7-way 分类", cx(sb), DY + 30, anchor="middle", fs=8.5, bold=True, fill="#2e7d32", family=FONT_EN)
for i, emo in enumerate(["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]):
    chip(SX + 8 + (i % 2) * 74, DY + 38 + (i // 2) * 19, 58, emo, english=True)
softmax_bar(SX + 12, DY + 122, 138, 14)
T("7类概率 · VA回归", cx(sb), DY + 152, anchor="middle", fs=9)
arl(ob, sb, color="#388e3c", marker="mkg")

# AP row — detailed protocol cards (目的 / 方法 / 结果 / 备注)
APY = MY + MH + 20
T("实验协议 AP0–AP4（三数据集混合 CREMA + MELD + MOSEI）", VW / 2, APY, anchor="middle", fs=11.5, bold=True, fill="#2e7d32")
aps = [
    ("AP0·三混合基线", [
        ("目的", "建立混合数据集基准性能"),
        ("方法", "standard 融合 + 三混合 50 epoch"),
        ("结果", "F1≈0.52（对照下界）"),
        ("状态", "固定融合策略"),
    ], False),
    ("AP1·单域上界", [
        ("目的", "探测单数据集性能上界"),
        ("方法", "CREMA / MELD / MOSEI 分别训练"),
        ("结果", "MELD F1≈0.54；MOSEI Acc≈0.72"),
        ("备注", "MELD 单域 → 部署优先 preset"),
    ], False),
    ("AP2★·三混合主实验", [
        ("目的", "Emotion Shift 主配置验证"),
        ("方法", "emotion_shift + M1–M4 配方消融"),
        ("结果", "F1≈0.562（当前最佳）"),
        ("备注", "ap2_m1 preset 部署候选"),
    ], True),
    ("AP3·融合策略消融", [
        ("目的", "对比四种 fusion 模块"),
        ("方法", "仅换 fusion_strategy，其余相同"),
        ("发现", "ES > LF > standard"),
        ("备注", "two_stage 不稳定易塌缩"),
    ], False),
    ("AP4·域适应微调", [
        ("目的", "缓解跨数据集域偏移"),
        ("方法", "DANN 域对抗 + λ 权重扫描"),
        ("结果", "λ=0.05 时 F1≈0.528"),
        ("备注", "未超 AP2 峰值 · 补充证据"),
    ], False),
]
apw, aph, apgap, apy = 368, 108, 16, APY + 18
apb = []
for i, (ti, rows, hi) in enumerate(aps):
    ax = 24 + i * (apw + apgap)
    ab = ap_card(ax, apy, apw, aph, ti, rows, highlight=hi)
    apb.append(ab)
for i in range(4):
    arl(apb[i], apb[i + 1])

CHK_Y = apy + aph + 14
B(24, CHK_Y, VW - 48, 26, "#e8f5e9", "#388e3c", 1.2, 3)
T("产出最优模型权重 → 加载至在线智能体情绪推理服务", VW / 2, CHK_Y + 13, anchor="middle", fs=10, bold=True)
A_BOTTOM = CHK_Y + 26 + A_PAD
bg(14, AY, VW - 28, A_BOTTOM - AY)
# F1 feedback: horizontal arrow along green checkpoint bar
FB_Y = CHK_Y + 13
apath(f"M{cx(ob)} {FB_Y} L{cx(eb)} {FB_Y}", color="#00897b", marker="mkg")
T("验证 F1 反馈 → 更新参数", (cx(ob) + cx(eb)) / 2, FB_Y - 14, anchor="middle", fs=9, bold=True, fill="#00897b")

# ═══════════════════ (b) ONLINE ═══════════════════
BY = A_BOTTOM + GAP
T("(b)", 24, BY + 18, fs=13, bold=True)
T("在线智能体：表现层 → 编排层 → 微服务层 · 六步推理流水线", 52, BY + 18, fs=11, bold=True)

FY = BY + 36
fb = B(24, FY, 280, 268, "#fafafa", "#555", 1.2, 3, "5,3")
T("表现层 · 浏览器客户端", cx(fb), FY + 18, anchor="middle", fs=11, bold=True)
icon("ico-react", 36, FY + 30, 18)
T("React 18", 58, FY + 39, fs=10, bold=True)
icon("ico-vite", 36, FY + 50, 18)
T("Vite 前端框架", 58, FY + 59, fs=10, bold=True)
for i, t in enumerate(["摄像头 / 麦克风采集", "分块上传 · WebSocket", "流水线状态监控", "七类情绪概率展示", "模型权重预设切换", "运行日志与结果"]):
    T(t, 36, FY + 78 + i * 18, fs=9.5, bold=True)
icon("ico-nginx", 36, FY + 198)
T("nginx HTTPS · Cloudflare", 56, FY + 206, fs=9.5, bold=True)

OX2 = 320
orb = B(OX2, FY, 600, 268, "#dce8f5", "#3d6ea8", 1.5, 3)
icon("ico-fastapi", OX2 + 12, FY + 8, 20)
T("编排层 · FastAPI :8000", OX2 + 38, FY + 18, fs=12, bold=True, fill="#1a4070")
T("请求接入 · 会话管理 · 结果返回", OX2 + 12, FY + 36, fs=9.5, bold=True)
for i, (n, d) in enumerate([
    ("模型路由", "加载训练权重"), ("推理适配", "桥接离线模型"), ("语音识别", "→ Whisper"),
    ("大模型对话", "→ Ollama"), ("分块重组", "流式上传"), ("滑窗缓冲", "3s/1s"),
    ("多模态仲裁", "最终标签"), ("ASR 校准", "语义修正"),
]):
    mx = OX2 + 12 + (i % 4) * 144
    my = FY + 52 + (i // 4) * 44
    mb2 = B(mx, my, 132, 38, "#fff", "#888", 1, 2)
    T(n, cx(mb2), my + 13, anchor="middle", fs=9, bold=True)
    T(d, cx(mb2), my + 27, anchor="middle", fs=8, fill="#444")

T("在线推理六步流水线", cx(orb), FY + 148, anchor="middle", fs=10, bold=True)
steps, step_bs = ["1接入", "2转写", "3合并", "4推理", "5校准", "6仲裁", "7回复"], []
for i, st in enumerate(steps):
    sx = OX2 + 12 + i * 82
    sy = FY + 164
    fill = "#fff8e1" if st[0] in "56" else ("#e8f5e9" if st[0] == "7" else "#fff")
    stroke = "#f9a825" if st[0] in "56" else ("#388e3c" if st[0] == "7" else "#555")
    sb2 = B(sx, sy, 72, 26, fill, stroke, 1, 13)
    step_bs.append(sb2)
    T(st, cx(sb2), cy(sb2), anchor="middle", fs=9, bold=True)
    if i:
        arl(step_bs[i - 1], sb2)

T("部署权重方案", OX2 + 12, FY + 214, fs=9, bold=True)
for i, p in enumerate(["MELD单域★", "三混合", "中文微调", "MOSEI", "域适应"]):
    chip(OX2 + 100 + i * 78, FY + 232, 72, p, "#e8f5e9" if "★" in p else "#fff")
arl(fb, orb)

MSX = 940
asr = B(MSX, FY, 220, 100, "#fff8ee", "#c8a020", 1.3, 3)
icon("ico-whisper", MSX + 12, FY + 8, 20)
T("ASR 微服务 :9010", MSX + 38, FY + 18, fs=11, bold=True)
T("faster-whisper · 中文", MSX + 12, FY + 38, fs=9.5, bold=True)
T("GPU 加速转写", MSX + 12, FY + 56, fs=9.5, bold=True)
T("语音 → 文本", MSX + 12, FY + 74, fs=9.5, bold=True)

GY = FY + 112
gpu = B(MSX, GY, 220, 108, "#e8f5e9", "#388e3c", 1.3, 3)
icon("ico-cuda", MSX + 12, GY + 8, 20)
icon("ico-pytorch", MSX + 36, GY + 8, 20)
T("情绪推理 GPU", MSX + 12, GY + 28, fs=11, bold=True)
T("ResNet + Wav2Vec + BERT", MSX + 12, GY + 48, fs=9.5, bold=True)
T("Emotion Shift 前向", MSX + 12, GY + 66, fs=9.5, bold=True)
T("多窗时序聚合", MSX + 12, GY + 84, fs=9.5, bold=True)

LY = GY + 120
llm = B(MSX, LY, 220, 96, "#f3eef8", "#7b5ea7", 1.3, 3)
icon("ico-ollama", MSX + 12, LY + 8, 20)
T("LLM 微服务 :11434", MSX + 38, LY + 18, fs=11, bold=True)
T("Ollama · Qwen2.5", MSX + 12, LY + 38, fs=9.5, bold=True)
T("共情对话 · 情绪注入", MSX + 12, LY + 56, fs=9.5, bold=True)

arl(orb, asr)
apath(
    f"M{right(step_bs[1])} {cy(step_bs[1])} L{right(step_bs[1]) + 16} {cy(step_bs[1])} "
    f"L{right(step_bs[1]) + 16} {FY + 50} L{MSX} {FY + 50}",
    color="#c8a020",
)
apath(
    f"M{right(step_bs[3])} {cy(step_bs[3])} L{right(step_bs[3]) + 16} {cy(step_bs[3])} "
    f"L{right(step_bs[3]) + 16} {GY + 46} L{MSX} {GY + 46}",
    color="#2e7d32",
    marker="mkg",
)
apath(
    f"M{right(step_bs[6])} {cy(step_bs[6])} L{right(step_bs[6]) + 16} {cy(step_bs[6])} "
    f"L{right(step_bs[6]) + 16} {LY + 38} L{MSX} {LY + 38}",
    color="#7b5ea7",
)

DX2 = 1180
dep = B(DX2, FY, 220, 100, "#fff", "#888", 1, 3)
icon("ico-nginx", DX2 + 12, FY + 8, 20)
T("部署与运维", DX2 + 38, FY + 18, fs=11, bold=True)
for i, t in enumerate(["HTTPS 反向代理", "进程守护 · tmux", "内网穿透演示", "多环境配置"]):
    T(t, DX2 + 12, FY + 38 + i * 16, fs=9.5, bold=True)
rt = B(DX2, GY, 220, 108, "#fff", "#888", 1, 3)
T("推理运行时", cx(rt), GY + 18, anchor="middle", fs=11, bold=True)
for i, t in enumerate(["真实 GPU 推理", "多窗时序聚合", "中英文 tokenizer", "WebSocket 推送"]):
    T(t, DX2 + 12, GY + 38 + i * 16, fs=9.5, bold=True)
cfg = B(DX2, LY, 220, 96, "#fff", "#888", 1, 3)
T("关键配置", cx(cfg), LY + 18, anchor="middle", fs=11, bold=True)
for i, t in enumerate(["模型权重方案", "ASR 引擎", "LLM 引擎", "GPU 设备"]):
    T(t, DX2 + 12, LY + 38 + i * 16, fs=9.5, bold=True)

B_BOTTOM = max(bottom(fb), bottom(llm), bottom(orb)) + B_PAD
bg(14, BY, VW - 28, B_BOTTOM - BY)
apath(
    f"M{cx(llm)} {bottom(llm)} Q{700} {B_BOTTOM - 6} {cx(fb)} {bottom(fb)}",
    color="#7b5ea7",
    dash="5,3",
)
T("共情回复与情绪结果 → 浏览器展示", 700, B_BOTTOM, anchor="middle", fs=9, bold=True, fill="#7b5ea7")
av(VW / 2, A_BOTTOM, BY, dash="4,3")

# ═══════════════════ (c) EXAMPLE ═══════════════════
CY = B_BOTTOM + GAP
bg(14, CY, VW - 28, C_H)
T("(c)", 24, CY + 18, fs=13, bold=True)
T("推理示例：用户说「哈哈哈哈」· 六步流水线逐步流转", 52, CY + 18, fs=11, bold=True)

snaps = [
    ("环境", "采集音视频", "—"), ("智能体", "Whisper 转写", "哈哈…"), ("智能体", "文本融合", "中文语义"),
    ("智能体", "GPU 三模态", "neutral↑"), ("环境", "ASR 校准", "happy↑"), ("智能体", "多模态仲裁", "happy"),
    ("智能体", "Qwen 回复", "共情话术"), ("界面", "结果展示", "概率+话术"),
]
bx, snap_bs = 32, []
for i, (r, a, v) in enumerate(snaps):
    sb3 = B(bx, CY + 36, 80, 54, "#fff", "#888", 1, 2)
    snap_bs.append(sb3)
    T(r, cx(sb3), CY + 48, anchor="middle", fs=9, bold=True)
    T(a, cx(sb3), CY + 62, anchor="middle", fs=8.5)
    T(v, cx(sb3), CY + 76, anchor="middle", fs=8, fill="#444")
    if i:
        arl(snap_bs[i - 1], sb3)
    bx += 86
for i, d in enumerate([
    "① 接入：浏览器采集视频与音频",
    "② 转写：Whisper 识别「哈哈哈哈」",
    "③ 合并：用户输入与语音文本融合",
    "④ 推理：三模态编码 → Emotion Shift 融合",
    "⑤ 校准：笑声语义提升 happy 概率",
    "⑥ 仲裁 + 回复：确定开心 → LLM 共情话术",
]):
    T(d, 820, CY + 40 + i * 18, fs=9.5, bold=True)

VIEW_H = CY + C_H + 16
svg = build_svg(VIEW_H)
OUT_SVG.write_text(svg, encoding="utf-8")
print(f"SVG  {OUT_SVG}  viewBox={VW}x{VIEW_H}")

png, jpg = export_raster(OUT_SVG, OUT_PNG, OUT_JPG, scale=2.0)
print(f"PNG  {png}  ({png.stat().st_size // 1024} KB)")
print(f"JPG  {jpg}  ({jpg.stat().st_size // 1024} KB)")
