# streamlit_chess_analyzer_full_features.py
"""
Streamlit Chess Analyzer — Full Features
- Full controls (Play / Pause / Next / Prev / Reset)
- Color-coded arrows (blunder/mistake/inaccuracy/good)
- Per-move evaluation text
- Engine score graph (matplotlib)
- Heatmap of mover-relative deltas (2 x N)
- Stockfish engine if available (configurable depth)
- Fallback material evaluation otherwise
- Robust SAN/PGN parsing, illegal move reporting
- Graceful fallbacks for optional libs (cairosvg)
"""

import re
import time
from io import BytesIO
import base64
from typing import List, Tuple, Dict, Any

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Optional libs handled gracefully
try:
    import chess
    import chess.svg
    import chess.engine
    CHESS_AVAILABLE = True
except Exception:
    chess = None
    chess_svg = None
    chess_engine = None
    CHESS_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

# -------------------------
# Utility functions
# -------------------------
def svg_to_png_bytes(svg_text: str):
    if not CAIROSVG_AVAILABLE:
        return None
    try:
        return cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
    except Exception:
        return None

# -------------------------
# Move extraction
# -------------------------
MOVE_REGEX = re.compile(
    r'\b('
    r'O-O-O|O-O|'                               # castling
    r'[KQNBR]?[a-h]?[1-8]?x?[a-h][1-8]'         # SAN move w/ optional capture
    r'(?:=[KQNBR])?'                            # promotion
    r'[+#]?'
    r')\b'
)

def extract_moves(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("[") and ln.endswith("]"):
            continue
        cleaned.append(ln)
    moves_str = " ".join(cleaned)
    moves_str = re.sub(r'\d+\.(\.\.)?', ' ', moves_str)
    moves_str = re.sub(r'\$\d+', ' ', moves_str)
    tokens = MOVE_REGEX.findall(moves_str)
    return [t.strip() for t in tokens if t and not t.isspace()]

# -------------------------
# Engine wrapper
# -------------------------
class EngineWrapper:
    def __init__(self, engine_path: str = "stockfish", depth: int = 12):
        self.depth = depth
        self.engine = None
        self.using_engine = False
        if not CHESS_AVAILABLE:
            return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            self.using_engine = True
        except Exception:
            self.engine = None
            self.using_engine = False

    def close(self):
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass

    def eval_cp(self, board) -> int:
        """Return centipawn evaluation (White POV) or None if engine failed."""
        if not CHESS_AVAILABLE:
            return None
        if self.using_engine and self.engine:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                score = info.get("score")
                if score is None:
                    return None
                sc = score.white()
                mate = sc.mate()
                if mate is not None:
                    return 100000 if mate > 0 else -100000
                cp = sc.score()
                return cp if cp is not None else None
            except Exception:
                return None
        return material_balance(board)

# -------------------------
# Material fallback
# -------------------------
def material_balance(board) -> int:
    if not CHESS_AVAILABLE or board is None:
        return 0
    vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    w = 0
    b = 0
    for ptype, v in vals.items():
        w += len(board.pieces(ptype, chess.WHITE)) * v
        b += len(board.pieces(ptype, chess.BLACK)) * v
    return (w - b) * 100

# -------------------------
# Analyze game
# -------------------------
def analyze_game(moves: List[str], engine_wrapper: EngineWrapper) -> Tuple[List[Dict[str,Any]], List[Tuple[int,str,str]], Any]:
    """
    Returns analysis_list, illegal_moves, final_board
    analysis_list elements:
      ply (1-based), san, move_obj, fen, eval (cp), is_capture (bool), gives_check (bool), material_before, material_after
    """
    if not CHESS_AVAILABLE:
        return [], [], None

    board = chess.Board()
    analysis = []
    illegal = []
    material_prev = material_balance(board)

    for ply, san in enumerate(moves, start=1):
        try:
            move = board.parse_san(san)
        except Exception as e:
            illegal.append((ply, san, str(e)))
            continue
        is_cap = board.is_capture(move)
        board.push(move)
        gives_check = board.is_check()
        material_after = material_balance(board)
        eval_cp = None
        try:
            eval_cp = engine_wrapper.eval_cp(board)
        except Exception:
            eval_cp = None
        if eval_cp is None:
            eval_cp = material_after
        analysis.append({
            "ply": ply,
            "san": san,
            "move_obj": move,
            "fen": board.fen(),
            "eval": int(round(eval_cp)),
            "is_capture": is_cap,
            "gives_check": gives_check,
            "material_before": material_prev,
            "material_after": material_after
        })
        material_prev = material_after

    return analysis, illegal, board

# -------------------------
# Classify mistakes + reason sentences
# -------------------------
def classify_and_reason(analysis: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    prev_eval = None
    for entry in analysis:
        cur_eval = entry["eval"]
        if prev_eval is None:
            prev_eval = 0
        delta = cur_eval - prev_eval
        mover = "White" if (entry["ply"] % 2 == 1) else "Black"
        mover_delta = delta if mover == "White" else -delta
        label = None
        if mover_delta <= -200:
            label = "Blunder"
        elif mover_delta <= -80:
            label = "Mistake"
        elif mover_delta <= -30:
            label = "Inaccuracy"
        if label:
            # build reason
            reasons = []
            mat_diff = entry["material_after"] - entry["material_before"]
            if mat_diff < 0:
                # material values scaled by *100; convert to pawn-equivalents roughly
                pawn_eq = abs(mat_diff) / 100.0
                reasons.append(f"lost {pawn_eq:.1f} pawn-equivalents")
            if entry["is_capture"]:
                reasons.append("it was a capture")
            if entry["gives_check"]:
                reasons.append("it gave check")
            reasons.append(f"eval changed by {int(round(mover_delta))} cp against {mover}")
            reason = ". ".join(reasons).strip() + "."
            # Capitalize first letter
            reason = reason[0].upper() + reason[1:]
            out.append({
                "ply": entry["ply"],
                "san": entry["san"],
                "mover": mover,
                "delta": int(round(mover_delta)),
                "label": label,
                "reason": reason
            })
        prev_eval = cur_eval
    return out

# -------------------------
# Color map for arrows and markers
# -------------------------
COLOR_MAP = {
    "Blunder": "#ff0000",      # red
    "Mistake": "#ff8800",      # orange
    "Inaccuracy": "#ffea00",   # yellow
    "Good": "#2ecc71",         # green
    "Default": "#888888"       # grey
}

def label_to_arrow_color(label: str) -> str:
    if not label:
        return COLOR_MAP["Default"]
    return COLOR_MAP.get(label, COLOR_MAP["Default"])

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Chess Analyzer — Full Features", layout="wide")
st.title("♟️ Chess Analyzer — Full Features (Arrows, Eval Graph, Heatmap, Reasons)")

# Warn about missing dependencies
if not CHESS_AVAILABLE:
    st.error("python-chess not found. Install with `pip install python-chess` and rerun.")
    st.stop()
if not CAIROSVG_AVAILABLE:
    st.warning("cairosvg not found — PNG conversion is disabled. Install with `pip install cairosvg` for best visuals.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    engine_path = st.text_input("Stockfish binary path (or 'stockfish')", value="stockfish")
    engine_depth = st.slider("Engine depth (if available)", 6, 28, 14)
    anim_delay = st.slider("Animation delay (s)", 0.05, 1.5, 0.45, step=0.05)
    show_graph = st.checkbox("Show evaluation graph", value=True)
    show_heatmap = st.checkbox("Show heatmap (mover deltas)", value=True)

# Input PGN / moves
st.subheader("Paste PGN / SAN moves")
pgn_text = st.text_area("PGN or SAN moves", height=300, placeholder="Paste PGN or moves here (tags ignored)")

moves = extract_moves(pgn_text)
st.write(f"Detected moves: {len(moves)}")
if len(moves) > 0:
    st.write("First moves:", moves[:20])

# Analyze
if st.button("Analyze"):
    ew = EngineWrapper(engine_path, engine_depth)
    analysis, illegal_moves, final_board = analyze_game(moves, ew)
    # annotate eval_type placeholder (will set below)
    st.session_state.analysis = analysis
    st.session_state.illegal_moves = illegal_moves
    st.session_state.final_board = final_board
    st.session_state.engine_wrapper = ew
    st.session_state.frame_index = 0
    st.session_state.playing = False

# Load existing state
analysis = st.session_state.get("analysis", None)
illegal_moves = st.session_state.get("illegal_moves", [])
final_board = st.session_state.get("final_board", None)
engine_wrapper = st.session_state.get("engine_wrapper", None)

if analysis is None:
    st.info("Press Analyze to parse and analyze the pasted moves.")
    st.stop()

# Show illegal moves
if illegal_moves:
    st.subheader("Illegal / Unparsed moves")
    for ply, san, err in illegal_moves:
        st.markdown(f"- Ply {ply}: `{san}` → _{err}_")

# Classify mistakes & reasons
mistakes = classify_and_reason(analysis)
# Attach a short label to each analysis entry for arrow coloring
for entry in analysis:
    # default Good if not flagged; we'll override for labeled plies
    entry["label"] = None
for m in mistakes:
    # find matching entry
    idx = m["ply"] - 1
    if 0 <= idx < len(analysis):
        analysis[idx]["label"] = m["label"]
        analysis[idx]["reason"] = m["reason"]

# Some entries without label we mark as Good if eval improved for mover
prev_eval = 0
for entry in analysis:
    mover = "White" if (entry["ply"] % 2 == 1) else "Black"
    cur_eval = entry["eval"]
    delta = cur_eval - prev_eval
    mover_delta = delta if mover == "White" else -delta
    if entry.get("label") is None:
        if mover_delta >= 30:
            entry["label"] = "Good"
    prev_eval = cur_eval

# Summary
st.subheader("Summary")
st.write(f"Plies analyzed: {len(analysis)} — Illegal moves: {len(illegal_moves)}")
if mistakes:
    st.markdown(f"**Detected {len(mistakes)} critical moves (blunder/mistake/inaccuracy).**")
else:
    st.success("No major mistakes detected by the heuristic.")

# Evaluation graph
if show_graph:
    st.subheader("Engine / Eval Graph (White POV)")
    xs = [e["ply"] for e in analysis]
    ys = [e["eval"] for e in analysis]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(xs, ys, marker='o', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.7)
    ax.set_xlabel("Ply")
    ax.set_ylabel("Eval (centipawn, White POV)")
    ax.grid(True, linestyle="--", alpha=0.4)
    # mark mistakes
    for m in mistakes:
        mply = m["ply"]
        yval = next((e["eval"] for e in analysis if e["ply"] == mply), 0)
        color = 'red' if m["label"] == "Blunder" else ('orange' if m["label"] == "Mistake" else 'yellow')
        ax.scatter(mply, yval, s=120, color=color, edgecolor='k', zorder=5)
        ax.annotate(f"{m['label']} ({m['mover']})\nΔ{m['delta']}cp",
                    xy=(mply, yval), xytext=(mply+0.1, yval+40),
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    st.pyplot(fig)

# Heatmap: mover-relative deltas (2 x N)
if show_heatmap:
    st.subheader("Heatmap — mover-relative deltas (row: White/Black)")
    if not analysis:
        st.write("No data")
    else:
        N = len(analysis)
        heat = np.zeros((2, N))
        prev = 0
        for i, e in enumerate(analysis):
            cur = e["eval"]
            if i == 0:
                prev = 0
            delta = cur - prev
            mover = 0 if ((e["ply"] % 2) == 1) else 1
            mover_delta = delta if mover == 0 else -delta
            heat[mover, i] = mover_delta
            prev = cur
        # Normalize for color scale
        vmax = np.max(np.abs(heat)) if np.any(heat) else 1
        fig2, ax2 = plt.subplots(figsize=(9, 2))
        cax = ax2.imshow(heat, aspect='auto', cmap='RdYlGn_r', vmin=-vmax, vmax=vmax)
        ax2.set_yticks([0,1])
        ax2.set_yticklabels(['White', 'Black'])
        ax2.set_xlabel("Ply index (0-based column per ply)")
        fig2.colorbar(cax, orientation='vertical', label='Mover-relative delta (cp)')
        st.pyplot(fig2)

# -------------------------
# Replay controls (session-state stable)
# -------------------------
st.subheader("Replay Controls (Play / Pause / Prev / Next / Reset)")
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
with c1:
    if st.button("⏮ Prev"):
        st.session_state.playing = False
        st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
with c2:
    if st.button("▶ Play"):
        st.session_state.playing = True
with c3:
    if st.button("⏸ Pause"):
        st.session_state.playing = False
with c4:
    if st.button("⏭ Next"):
        st.session_state.playing = False
        st.session_state.frame_index = min(len(analysis), st.session_state.frame_index + 1)
with c5:
    if st.button("⟲ Reset"):
        st.session_state.playing = False
        st.session_state.frame_index = 0

# Slider scrubber
max_index = len(analysis)
new_index = st.slider("Scrub to ply (0 = start)", 0, max_index, st.session_state.frame_index)
if new_index != st.session_state.frame_index:
    st.session_state.frame_index = new_index
    st.session_state.playing = False

# Build board to current frame
board = chess.Board()
for i in range(st.session_state.frame_index):
    board.push(analysis[i]["move_obj"])

# Determine last move and arrow color
last_move = None
arrows = None
if st.session_state.frame_index > 0:
    last_entry = analysis[st.session_state.frame_index - 1]
    last_move = last_entry["move_obj"]
    label = last_entry.get("label", None)
    arrow_color = label_to_arrow_color(label)
    try:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        # chess.svg.Arrow takes squares (0..63) and optional color
        arrows = [chess.svg.Arrow(from_sq, to_sq, color=arrow_color)]
    except Exception:
        arrows = None

# Render board (with lastmove highlight and arrows)
try:
    svg = chess.svg.board(board=board, size=640, lastmove=last_move, arrows=arrows)
except Exception:
    try:
        svg = chess.svg.board(board=board, size=640, lastmove=last_move)
    except Exception:
        svg = chess.svg.board(board=board, size=640)

png = svg_to_png_bytes(svg)
if png:
    st.image(png)
else:
    # raw svg fallback
    st.components.v1.html(svg, height=680)

# Per-move evaluation text / reason
if st.session_state.frame_index == 0:
    st.write("Initial position (before any moves).")
else:
    le = analysis[st.session_state.frame_index - 1]
    mover = "White" if (le["ply"] % 2 == 1) else "Black"
    st.write(f"After ply {le['ply']}: **{mover}** played `{le['san']}` — Eval: {le['eval']} cp (White POV)")
    # show reason if classified
    reason = le.get("reason", None)
    label = le.get("label", None)
    if label:
        st.markdown(f"**{label}** for **{mover}** — {reason}")

# Auto-advance when playing
if st.session_state.playing:
    if st.session_state.frame_index < max_index:
        st.session_state.frame_index += 1
        time.sleep(anim_delay)
        st.experimental_rerun()
    else:
        st.session_state.playing = False
        st.session_state.frame_index = max_index

# Final position display
st.subheader("Final Position")
try:
    fsvg = chess.svg.board(board=final_board, size=520)
except Exception:
    fsvg = chess.svg.board(board=final_board, size=520)
fpng = svg_to_png_bytes(fsvg)
if fpng:
    st.image(fpng)
else:
    st.components.v1.html(fsvg, height=520)

# Attempt to close engine cleanly
def _close_engine():
    ew = st.session_state.get("engine_wrapper", None)
    if ew:
        try:
            ew.close()
        except Exception:
            pass

_close_engine()