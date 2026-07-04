#!/usr/bin/env python3
"""Frame-by-frame video translator using ffmpeg filters.

Approach: OCR sampled frames → Translate → Build ffmpeg filter chain
(drawbox to cover Japanese + drawtext to overlay English) → Single pass render.

Features:
- Text enlargement for small UI text: crop → upscale → enhance → re-OCR
- UI label dictionary for known Clip Studio Paint labels
- Hybrid OCR: EasyOCR detection + manga-ocr recognition on enlarged crops
- Deduplication of same text at consecutive timestamps
- Background color sampling for seamless inpainting
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("translate_video")

# ── Configuration ──────────────────────────────────────────────────────────
FONT_PATH = None  # auto-detect
FONT_SIZE_MIN = 12
FONT_SIZE_MAX = 48
INPAINT_PADDING = 4
MIN_CONFIDENCE = 0.25
MIN_TEXT_LEN = 2
MIN_BBOX_AREA = 300  # lowered to capture small UI text
CANVAS_CONFIDENCE = 0.05  # lower threshold for CANVAS region text

# Text enlargement settings
ENLARGE_THRESHOLD_WIDTH = 150   # pixels - text narrower than this gets enlarged
ENLARGE_THRESHOLD_HEIGHT = 40   # pixels - text shorter than this gets enlarged
ENLARGE_FACTOR = 3              # how much to enlarge small text
ENHANCE_SHARPNESS = 2.0
ENHANCE_CONTRAST = 1.5

# ── UI Label Dictionary ────────────────────────────────────────────────────
# Known Clip Studio Paint UI labels with their English translations.
# These are used directly instead of relying on OCR + Google Translate.
UI_LABEL_DICT = {
    # Menu bar items
    "ファイル": "File",
    "編集": "Edit",
    "表示": "View",
    "選択": "Selection",
    "フィルタ": "Filter",
    "ウインドウ": "Window",
    "ヘルプ": "Help",
    "アニメーション": "Animation",
    "ツール": "Tool",
    "3D": "3D",
    "お問い合わせ": "Contact",

    # File menu
    "新規": "New",
    "開く": "Open",
    "保存": "Save",
    "名前を付けて保存": "Save As",
    "書き出し": "Export",
    "環境設定": "Preferences",
    "設定": "Settings",
    "環境": "Environment",
    "ショートカット": "Shortcuts",
    "作業フォルダ": "Work Folder",
    "ページ管理": "Page Management",

    # Edit menu
    "元に戻す": "Undo",
    "戻る": "Undo",
    "やり直し": "Redo",
    "進む": "Redo",
    "切り取り": "Cut",
    "コピー": "Copy",
    "貼り付け": "Paste",
    "削除": "Delete",
    "全て選択": "Select All",
    "選択範囲を変更": "Modify Selection",
    "選択範囲を反転": "Invert Selection",
    "選択解除": "Deselect",
    "塗りつぶし": "Fill",

    # View menu
    "拡大縮小": "Zoom",
    "回転": "Rotate",
    "反転": "Flip",
    "グリッド": "Grid",
    "定規": "Ruler",
    "ガイド": "Guide",

    # Tools
    "ペンツール": "Pen Tool",
    "ベンツール": "Pen Tool",
    "ブラシ": "Brush",
    "消しゴム": "Eraser",
    "イージーステイン": "Blend",
    "バケツツール": "Bucket Tool",
    "グラデーション": "Gradient",
    "選択ペン": "Selection Pen",
    "選択消しゴム": "Selection Eraser",
    "自動選択": "Auto Select",
    "投げ縄": "Lasso",
    "矩形選択": "Rectangular Select",
    "楕円選択": "Ellipse Select",
    "テキストツール": "Text Tool",
    "ものさし": "Ruler Tool",
    "図形": "Shape",
    "填充": "Fill",
    "操作ツール": "Object Tool",

    # Tool options
    "ツールオプション": "Tool Options",
    "オプション": "Options",
    "サブツール": "Sub Tool",
    "サブツール詳細": "Sub Tool Detail",
    "ブラシサイズ": "Brush Size",
    "サイズ": "Size",
    "不透明度": "Opacity",
    "硬度": "Hardness",
    "濃度": "Density",
    "描画色": "Drawing Color",
    "色選択": "Color Selection",
    "線幅変更": "Line Width",
    "太さ": "Thickness",
    "角度": "Angle",
    "間隔": "Spacing",
    "なめらかさ": "Smoothness",
    "手ぶれ補正": "Stabilization",
    "補正": "Correction",
    "強さ": "Strength",
    "最小幅": "Minimum Width",
    "最大幅": "Maximum Width",
    "ホイール": "Wheel",
    "インク": "Ink",
    "アンチエイリアス": "Anti-aliasing",

    # Layer panel
    "レイヤー": "Layer",
    "レイヤ": "Layer",
    "通常レイヤー": "Normal Layer",
    "フォルダー": "Folder",
    "フォルダ": "Folder",
    "新規レイヤー": "New Layer",
    "新規フォルダー": "New Folder",
    "レイヤーの結合": "Merge Layers",
    "統合": "Flatten",
    "表示レイヤーの統合": "Flatten Visible",
    "レイヤーの複製": "Duplicate Layer",
    "レイヤーの削除": "Delete Layer",
    "クリッピング": "Clipping",
    "マスク": "Mask",
    "レイヤーマスク": "Layer Mask",
    "ブレンドモード": "Blend Mode",
    "通常": "Normal",
    "乗算": "Multiply",
    "スクリーン": "Screen",
    "オーバーレイ": "Overlay",
    "比較（明）": "Lighten",
    "比較（暗）": "Darken",
    "差分": "Difference",
    "色相": "Hue",
    "彩度": "Saturation",
    "色相・彩度": "Hue/Saturation",
    "色彩調整": "Color Adjustment",
    "色相・彩度・明度": "Hue/Sat/Light",
    "明度": "Brightness",
    "コントラスト": "Contrast",
    "ガンマ": "Gamma",
    "レベル補正": "Levels",
    "トーンカーブ": "Tone Curve",
    "色温度": "Color Temperature",
    "色相環": "Color Wheel",
    "カラーパレット": "Color Palette",
    "カラーヒストグラム": "Color Histogram",

    # Canvas
    "キャンバス": "Canvas",
    "カンバス": "Canvas",
    "ページ": "Page",
    "作品": "Work",

    # Filters
    "ぼかし": "Blur",
    "シャープ": "Sharpen",
    "ノイズ": "Noise",
    "モザイク": "Mosaic",
    "変形": "Transform",
    "歪み": "Distort",
    "色変換": "Color Conversion",

    # Animation
    "セル": "Cell",
    "タイムライン": "Timeline",
    "フレーム": "Frame",
    "フレームレート": "Frame Rate",
    "再生": "Playback",
    "オニオンスキン": "Onion Skin",
    "キーフレーム": "Keyframe",
    "トラック": "Track",

    # Controller / misc
    "コントローラ": "Controller",
    "ナビゲータ": "Navigator",
    "情報": "Info",
    "ヒストリ": "History",
    "履歴": "History",
    "素材": "Material",
    "パース": "Perspective",
    "パース定規": "Perspective Ruler",
    "レースリール": "Racing Reel",  # specific UI tool name
    "カスタムツール": "Custom Tool",
    "ポーズスト": "Pose",
    "もじ": "Text",
    "文字": "Text",
    "サンプル": "Sample",
    "3Dレイヤー": "3D Layer",
    "ベクターレイヤー": "Vector Layer",
    "ラスターライヤー": "Raster Layer",

    # Common garbled OCR variants of UI labels
    "ベンツール": "Pen Tool",
    "ベンツールオプション": "Pen Tool Options",
    "ベンツールガション": "Pen Tool Options",
    "ベンソール": "Pen Tool",
    "ベンソールイプション": "Pen Tool Options",
    "ベンソールメンコン": "Pen Tool Options",
    "ペンソール": "Pen Tool",
    "ペンソールメプション": "Pen Tool Options",
    "ペンノールメンコン": "Pen Tool Options",
    "ハラー": "Halftone",
    "ハーン": "Halftone",
    "ハーフトーン": "Halftone",
    "0ハラー": "Halftone",
    "0リラー": "Layer",
    "アクシュン": "Action",
    "アキション": "Action",

    # Other common labels
    "新規作成": "Create New",
    "テンプレート": "Template",
    "用紙": "Paper",
    "キャンバスサイズ": "Canvas Size",
    "解像度": "Resolution",
    "背景色": "Background Color",
    "透明": "Transparent",
    "白": "White",
    "黒": "Black",
    "プレビュー": "Preview",
    "適用": "Apply",
    "キャンセル": "Cancel",
    "OK": "OK",
    "完了": "Done",
    "閉じる": "Close",
    "最小化": "Minimize",
    "最大化": "Maximize",
    "元に戻す": "Restore",
    "プロパティ": "Properties",
    "詳細": "Details",
    "一覧": "List",
    "検索": "Search",
    "お気に入り": "Favorites",
    "ダウンロード": "Download",
    "インポート": "Import",
    "エクスポート": "Export",
    "初期化": "Initialize",
    "リセット": "Reset",
    "反時計回り": "Counter-clockwise",
    "時計回り": "Clockwise",
    "上下反転": "Flip Vertical",
    "左右反転": "Flip Horizontal",
    "自由変形": "Free Transform",
    "拡大": "Enlarge",
    "縮小": "Shrink",
    "移動": "Move",
    "回転": "Rotate",
}


def _is_ui_label(text):
    """Check if text is a known UI label and return its translation.
    
    Uses multiple matching strategies:
    1. Direct match
    2. Text contains a known UI label as substring
    3. Fuzzy prefix match for garbled OCR of known labels
    """
    text_clean = text.strip()
    
    # 1. Direct match
    if text_clean in UI_LABEL_DICT:
        return UI_LABEL_DICT[text_clean]
    
    # 2. Text contains a known UI label as substring
    for jp, en in UI_LABEL_DICT.items():
        if jp in text_clean and len(text_clean) <= len(jp) + 6:
            return en
    
    # 3. Fuzzy match for garbled OCR of known labels
    # Common OCR errors: ソ/ン confusion, ツ/シ confusion, ク/ツ confusion
    # Try matching with normalized katakana
    def normalize_katakana(t):
        """Normalize commonly confused katakana characters."""
        return t.replace('ソ', 'ン').replace('ツ', 'シ').replace('ク', 'ツ').replace('ー', '').replace('・', '').replace('…', '').replace('。', '').replace('、', '').replace('.', '').replace(' ', '')
    
    text_norm = normalize_katakana(text_clean)
    for jp, en in UI_LABEL_DICT.items():
        jp_norm = normalize_katakana(jp)
        if len(jp_norm) >= 3 and text_norm == jp_norm:
            return en
        # Prefix match: if the normalized text starts with a normalized UI label
        if len(jp_norm) >= 4 and text_norm.startswith(jp_norm) and len(text_norm) <= len(jp_norm) + 4:
            return en
    
    # 4. For text with 。。。(dots), try stripping them and matching
    text_stripped = re.sub(r'[\.。\u2026\s]+', '', text_clean)
    if text_stripped in UI_LABEL_DICT:
        return UI_LABEL_DICT[text_stripped]
    for jp, en in UI_LABEL_DICT.items():
        if jp in text_stripped and len(text_stripped) <= len(jp) + 4:
            return en
    
    return None


def _is_garbled_translation(original, translated):
    """Check if a translation is likely garbled/nonsense.
    
    Signs of garbled translation:
    - Original was garbled OCR → translated to nonsense English
    - Translation contains unusual patterns
    - Very short katakana sequences that aren't real words
    """
    # If the original was properly matched by UI dict, it's not garbled
    if _is_ui_label(original) is not None:
        return False
    
    # If translation looks like nonsensical English (very short, unusual patterns)
    orig_stripped = original.strip()
    
    # Check for garbled katakana that got translated to nonsense
    katakana_only = re.sub(r'[\u30a1-\u30ff\uff66-\uff9f\u30fb\u30fc\u3000-\u303f\uff01-\uff5e\.。…、]', '', orig_stripped)
    if len(katakana_only) == 0 and len(orig_stripped) <= 10:
        # All-katakana short sequence - likely garbled tool name
        # Check if it looks like a known UI label after normalization
        if _is_ui_label(orig_stripped) is None:
            return True  # Likely garbled
    
    # Check for translations that look like transliterations of garbled text
    # e.g. "benzol iption", "Pennor Menkong", "Benthol Menkong"
    nonsensical_patterns = [
        r'^[A-Z][a-z]+ol\b',      # -ol endings (benzol, pensol)
        r'Menkon', r'Menkong',     # common garbled patterns
        r'iption$', r'meption$',   # from misread オプション/メプション
        r'pensol', r'benthol',     # garbled ペンソール, ベンソール
        r'haan$', r'Haller$',      # garbled ハーフトーン etc.
        r'lire$', r'hurrah$',      # garbled レイヤー, ブラシ
        r'Akshun', r'Axene',       # garbled アクション
    ]
    for pat in nonsensical_patterns:
        if re.search(pat, translated, re.IGNORECASE):
            return True
    
    # Check if the original text looks like garbled OCR of a known UI label
    # by checking if any UI label starts with the same first 2-3 characters
    orig_katakana = re.sub(r'[^\u30a1-\u30ff\uff66-\uff9f\u3040-\u309f\u4e00-\u9fff]', '', orig_stripped)
    if len(orig_katakana) >= 3:
        # Check if it starts like a known UI label but is garbled
        for jp in UI_LABEL_DICT:
            jp_katakana = re.sub(r'[^\u30a1-\u30ff\uff66-\uff9f\u3040-\u309f\u4e00-\u9fff]', '', jp)
            if len(jp_katakana) >= 3:
                # If the first 2 chars match but the rest is garbled
                if orig_katakana[:2] == jp_katakana[:2] and abs(len(orig_katakana) - len(jp_katakana)) <= 2:
                    # Check character similarity
                    same_chars = sum(1 for a, b in zip(orig_katakana, jp_katakana) if a == b)
                    if same_chars >= len(jp_katakana) * 0.4 and same_chars < len(jp_katakana) * 0.8:
                        return True  # Partial match = garbled version of this label
    
    return False


# ── Font detection ─────────────────────────────────────────────────────────
def find_cjk_font():
    """Find a CJK-capable font on the system."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, dirs, files in os.walk("/usr/share/fonts"):
        for f in files:
            fl = f.lower()
            if ("noto" in fl and "cjk" in fl) or "dejavu" in fl:
                if f.endswith((".ttf", ".ttc", ".otf")):
                    return os.path.join(root, f)
    return None


def install_cjk_font():
    """Install CJK fonts if not found."""
    logger.info("Installing Noto CJK font...")
    subprocess.run(["apt-get", "install", "-y", "fonts-noto-cjk"],
                    capture_output=True, timeout=120)
    subprocess.run(["apt-get", "install", "-y", "fonts-dejavu-core"],
                    capture_output=True, timeout=60)


# ── Video info ──────────────────────────────────────────────────────────────
def get_video_info(video_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        info = json.loads(result.stdout)
        vs = next((s for s in info.get("streams", [])
                    if s.get("codec_type") == "video"), {})
        fps_str = vs.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        return {
            "width": int(vs.get("width", 0)),
            "height": int(vs.get("height", 0)),
            "duration": float(info.get("format", {}).get("duration", 0)),
            "fps": fps,
        }
    except:
        return {}


# ── Frame extraction (for OCR only, small JPEGs) ───────────────────────────
def extract_ocr_frames(video_path, output_dir, interval_sec=2.0):
    """Extract frames at intervals for OCR. Returns list of (path, timestamp)."""
    duration = get_video_info(str(video_path)).get("duration", 0)
    if duration <= 0:
        duration = 600

    extract_fps = 1.0 / interval_sec
    pattern = str(Path(output_dir) / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={extract_fps:.4f}",
        "-q:v", "3",
        pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"Frame extraction failed: {result.stderr[-500:]}")
        return []

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    output = [(f, i * interval_sec) for i, f in enumerate(frames) if i * interval_sec <= duration]
    logger.info(f"Extracted {len(output)} OCR frames")
    return output


def extract_single_frame(video_path, timestamp, output_path):
    """Extract a single frame at original resolution for bg color sampling."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode == 0


# ── Smart frame dedup ───────────────────────────────────────────────────────
def deduplicate_frames(frame_list, sample_step=1):
    """Return indices of frames that differ from previous."""
    if len(frame_list) <= 10:
        return list(range(len(frame_list)))

    keep = [0]
    prev_hash = None

    for i in range(0, len(frame_list), sample_step):
        path = frame_list[i][0] if isinstance(frame_list[i], tuple) else frame_list[i]
        try:
            with Image.open(path) as img:
                thumb = img.resize((32, 32), Image.Resampling.BILINEAR).convert("L")
                h = hashlib.md5(thumb.tobytes()).hexdigest()
            if prev_hash is None or h != prev_hash:
                keep.append(i)
                prev_hash = h
        except:
            keep.append(i)

    if keep[-1] != len(frame_list) - 1:
        keep.append(len(frame_list) - 1)

    return sorted(set(keep))


# ── Region classification ──────────────────────────────────────────────────
def classify_region(bbox, img_w, img_h):
    cx = (min(p[0] for p in bbox) + max(p[0] for p in bbox)) / 2
    cy = (min(p[1] for p in bbox) + max(p[1] for p in bbox)) / 2
    if img_h > img_w * 1.5:
        if cx < img_w * 0.15: return "LEFT_TOOLBAR"
        if cx > img_w * 0.70: return "RIGHT_PANEL"
        if cy < img_h * 0.06: return "TOP_BAR"
        if cy > img_h * 0.94: return "BOTTOM_BAR"
        return "CANVAS"
    else:
        if cx < img_w * 0.12: return "LEFT_TOOLBAR"
        if cx > img_w * 0.78: return "RIGHT_PANEL"
        if cy < img_h * 0.08: return "TOP_BAR"
        if cy > img_h * 0.92: return "BOTTOM_BAR"
        return "CANVAS"


# ── Text enlargement ────────────────────────────────────────────────────────
def enlarge_and_enhance(image_crop, factor=ENLARGE_FACTOR):
    """Enlarge a text crop and enhance it for better OCR.

    Steps: upscale → grayscale → contrast → sharpen → threshold → RGB
    """
    new_w = image_crop.width * factor
    new_h = image_crop.height * factor
    enlarged = image_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    gray = enlarged.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(ENHANCE_CONTRAST)
    enhancer = ImageEnhance.Sharpness(gray)
    gray = enhancer.enhance(ENHANCE_SHARPNESS)

    # Threshold to binary (clean black text on white background)
    threshold = 128
    binary = gray.point(lambda x: 255 if x > threshold else 0, '1')
    result = binary.convert("RGB")
    return result


def should_enlarge(bbox_w, bbox_h, confidence):
    """Determine if a text region should be enlarged for better OCR."""
    if bbox_w < ENLARGE_THRESHOLD_WIDTH or bbox_h < ENLARGE_THRESHOLD_HEIGHT:
        return True
    if confidence < 0.6 and (bbox_w < ENLARGE_THRESHOLD_WIDTH * 2 or
                              bbox_h < ENLARGE_THRESHOLD_HEIGHT * 2):
        return True
    return False


# ── Garbage filter ──────────────────────────────────────────────────────────
def filter_garbage(results):
    """Remove likely garbage/noise entries from OCR results.

    Now less aggressive: UI labels are kept (they'll be translated via
    UI_LABEL_DICT or Google Translate). Only truly garbled OCR is removed.
    """
    filtered = []
    for r in results:
        text = r.get("text", "").strip()
        if not text:
            continue

        # Skip pure ellipsis/periods
        if re.match(r'^[\.。\u2026\s]+[!？！]*$', text):
            continue

        # Skip very short text with low confidence (but NOT if it's a known UI label)
        if len(text) <= 2 and r.get("confidence", 0) < 0.9:
            if _is_ui_label(text) is None:
                continue

        # Skip entries that are only punctuation and symbols
        if re.match(r'^[^\w\u3040-\u9fff]+$', text):
            continue

        # Check if it's a known UI label — always keep these
        if _is_ui_label(text) is not None:
            filtered.append(r)
            continue

        # Skip text with very low Japanese character ratio (garbled OCR)
        jp_chars = sum(1 for c in text if '\u3040' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
        total_chars = len(text.replace(' ', ''))
        if total_chars > 4 and jp_chars / total_chars < 0.3 and r.get("confidence", 0) < 0.4:
            continue

        # Skip garbled katakana sequences with low confidence
        katakana_only = re.sub(r'[\u30a1-\u30ff\uff66-\uff9f\u30fb\u30fc\u3000-\u303f\uff01-\uff5e]', '', text)
        if len(katakana_only) == 0 and len(text) <= 10 and r.get("confidence", 0) < 0.5:
            continue

        # Skip UI panel text with very low confidence (but higher threshold than before)
        region = r.get("region", "")
        if region in ("RIGHT_PANEL", "LEFT_TOOLBAR", "TOP_BAR", "BOTTOM_BAR"):
            if r.get("confidence", 0) < 0.35:
                continue

        filtered.append(r)
    return filtered


# ── OCR with enlargement ────────────────────────────────────────────────────
def run_ocr_on_frames(frame_paths, backend="manga_ocr", lang="ja",
                      video_path=None, orig_width=None, orig_height=None,
                      interval_sec=2.0):
    """Run OCR on selected frames with text enlargement for small UI text.

    Pipeline per frame:
    1. EasyOCR detection on full frame
    2. For each detected region:
       a. Classify region (CANVAS, LEFT_TOOLBAR, RIGHT_PANEL, etc.)
       b. If small/low-confidence: crop, enlarge 3x, enhance, re-OCR
       c. If large CANVAS text: try manga-ocr on cropped region
    3. For enlarged regions: try manga-ocr first, then EasyOCR on enlarged
    4. Apply garbage filter
    5. Deduplicate across timestamps
    """
    import easyocr

    logger.info("Initializing EasyOCR reader...")
    reader = easyocr.Reader(["ja", "en"], gpu=False, verbose=False)

    manga_engine = None
    if backend == "manga_ocr":
        try:
            from manga_ocr import MangaOcr
            logger.info("Initializing manga-ocr engine...")
            manga_engine = MangaOcr()
        except ImportError:
            logger.warning("manga-ocr not available, using EasyOCR only")
            backend = "easyocr"

    all_ocr = []
    ocr_width = None

    for idx, (frame_path, timestamp) in enumerate(frame_paths):
        logger.info(f"OCR frame {idx+1}/{len(frame_paths)}: ts={timestamp:.1f}s")

        with Image.open(frame_path) as img:
            frame_w, frame_h = img.size
            if ocr_width is None:
                ocr_width = frame_w

        scale_x = orig_width / frame_w if orig_width and frame_w else 1.0
        scale_y = orig_height / frame_h if orig_height and frame_h else 1.0

        # Step 1: EasyOCR detection
        try:
            easy_results = reader.readtext(str(frame_path), paragraph=False)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            continue

        frame_results = []

        # Open image once for all crops
        frame_image = Image.open(frame_path)

        for bbox, text, confidence in easy_results:
            if not text.strip() or len(text.strip()) < MIN_TEXT_LEN:
                continue

            bx1 = int(min(p[0] for p in bbox))
            by1 = int(min(p[1] for p in bbox))
            bx2 = int(max(p[0] for p in bbox))
            by2 = int(max(p[1] for p in bbox))
            bw = bx2 - bx1
            bh = by2 - by1
            barea = bw * bh

            # Classify region first
            temp_region = classify_region(
                [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                orig_width or frame_w, orig_height or frame_h
            )

            # Lower confidence threshold for CANVAS text
            min_conf = CANVAS_CONFIDENCE if temp_region == "CANVAS" and barea > 2000 else MIN_CONFIDENCE
            if confidence < min_conf:
                continue

            # Minimum bbox area (lowered for UI text)
            if barea < MIN_BBOX_AREA:
                continue

            # Scale bbox to original video resolution
            bbox_scaled = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in bbox]

            result = {
                "bbox": bbox_scaled,
                "text": text.strip(),
                "confidence": round(float(confidence), 3),
                "timestamp_sec": round(timestamp, 2),
                "region": temp_region,
                "enlarged": False,
            }

            # ── Text enlargement pipeline for small/low-confidence text ──
            if should_enlarge(bw, bh, confidence) and manga_engine is not None:
                try:
                    # Crop with padding
                    pad = max(5, int(min(bw, bh) * 0.1))
                    crop_box = (
                        max(0, bx1 - pad),
                        max(0, by1 - pad),
                        min(frame_image.width, bx2 + pad),
                        min(frame_image.height, by2 + pad),
                    )
                    crop = frame_image.crop(crop_box)

                    # Enlarge and enhance
                    enlarged = enlarge_and_enhance(crop)
                    result["enlarged"] = True

                    # Try manga-ocr on the enlarged crop (best for Japanese)
                    manga_text = manga_engine(enlarged)
                    if manga_text.strip() and not re.match(r'^[\.。\u2026\s]+$', manga_text):
                        # Validate: manga-ocr result should have decent Japanese char ratio
                        manga_jp = sum(1 for c in manga_text if '\u3040' <= c <= '\u9fff')
                        manga_total = len(manga_text.replace(' ', ''))
                        if manga_total > 0 and manga_jp / manga_total >= 0.3:
                            result["text"] = manga_text.strip()
                            result["confidence"] = max(result["confidence"], 0.75)
                            logger.debug(f"manga-ocr (enlarged): '{text.strip()}' -> '{manga_text.strip()}'")
                        else:
                            # manga-ocr returned non-Japanese garbage on enlarged crop
                            # Try EasyOCR on the enlarged crop
                            try:
                                enlarged_results = reader.readtext(enlarged, paragraph=False)
                                if enlarged_results:
                                    best = max(enlarged_results, key=lambda x: x[2])
                                    if best[2] > confidence and best[1].strip():
                                        result["text"] = best[1].strip()
                                        result["confidence"] = round(float(best[2]), 3)
                                        logger.debug(f"EasyOCR (enlarged): '{text.strip()}' -> '{best[1].strip()}'")
                            except:
                                pass
                    else:
                        # manga-ocr returned nothing useful on enlarged crop
                        # Try EasyOCR on enlarged crop
                        try:
                            enlarged_results = reader.readtext(enlarged, paragraph=False)
                            if enlarged_results:
                                best = max(enlarged_results, key=lambda x: x[2])
                                if best[2] > confidence and best[1].strip():
                                    result["text"] = best[1].strip()
                                    result["confidence"] = round(float(best[2]), 3)
                        except:
                            pass
                except Exception as e:
                    logger.debug(f"Enlargement pipeline failed: {e}")

            # ── manga-ocr refinement for large CANVAS text (no enlargement needed) ──
            elif manga_engine and backend == "manga_ocr" and temp_region == "CANVAS":
                try:
                    aspect = bw / max(1, bh)

                    if aspect >= 1.5:
                        # Good aspect ratio — run manga-ocr with padding
                        pad = max(5, int(min(bw, bh) * 0.08))
                        crop = frame_image.crop((
                            max(0, bx1 - pad), max(0, by1 - pad),
                            min(frame_image.width, bx2 + pad), min(frame_image.height, by2 + pad)
                        ))
                        refined = manga_engine(crop)

                        if refined.strip() and len(refined.strip()) >= len(text.strip()) * 0.5:
                            easy_jp = sum(1 for c in text if '\u3040' <= c <= '\u9fff')
                            manga_jp = sum(1 for c in refined if '\u3040' <= c <= '\u9fff')
                            if manga_jp >= easy_jp * 0.8:
                                result["text"] = refined.strip()
                                result["confidence"] = max(result["confidence"], 0.75)
                    else:
                        # Bad aspect ratio — try splitting into horizontal strips
                        strip_height = (by2 - by1) // 2
                        strips_text = []
                        for sy in range(by1, by2, max(strip_height, 20)):
                            sy_end = min(sy + strip_height + 5, by2)
                            strip_crop = frame_image.crop((
                                max(0, bx1 - 3), max(0, sy - 3),
                                min(frame_image.width, bx2 + 3), min(frame_image.height, sy_end + 3)
                            ))
                            strip_text = manga_engine(strip_crop)
                            if strip_text.strip() and not re.match(r'^[\.。\u2026\s]+$', strip_text):
                                strips_text.append(strip_text.strip())

                        if strips_text:
                            combined = ' '.join(strips_text)
                            if len(combined) >= len(text.strip()) * 0.5:
                                result["text"] = combined
                                result["confidence"] = max(result["confidence"], 0.75)
                except Exception as e:
                    logger.debug(f"manga-ocr refinement failed: {e}")

            frame_results.append(result)

        frame_image.close()

        if frame_results:
            frame_results = filter_garbage(frame_results)
            all_ocr.extend(frame_results)

    # Deduplicate
    all_ocr = deduplicate_ocr_results(all_ocr, interval_sec=2.0)

    logger.info(f"OCR found {len(all_ocr)} text regions total (after dedup)")
    return all_ocr


def deduplicate_ocr_results(results, interval_sec=2.0):
    """Merge same-text detections at similar positions into single entries."""
    if not results:
        return results

    results.sort(key=lambda r: r["timestamp_sec"])

    merged = []
    for r in results:
        found_merge = False
        for m in merged:
            same_text = r["text"] == m["text"]
            close_position = bbox_overlap(r["bbox"], m["bbox"]) > 0.3
            close_time = abs(r["timestamp_sec"] - m["timestamp_sec"]) < interval_sec * 3

            if same_text and close_position and close_time:
                m["timestamp_sec"] = min(m["timestamp_sec"], r["timestamp_sec"])
                m["end_sec"] = max(m.get("end_sec", m["timestamp_sec"] + interval_sec),
                                   r["timestamp_sec"] + interval_sec)
                m["confidence"] = max(m["confidence"], r["confidence"])
                found_merge = True
                break

        if not found_merge:
            r_copy = dict(r)
            r_copy["end_sec"] = r["timestamp_sec"] + interval_sec
            merged.append(r_copy)

    return merged


def bbox_overlap(bbox_a, bbox_b):
    """Compute IoU-like overlap between two bboxes."""
    try:
        ax1, ay1 = min(p[0] for p in bbox_a), min(p[1] for p in bbox_a)
        ax2, ay2 = max(p[0] for p in bbox_a), max(p[1] for p in bbox_a)
        bx1, by1 = min(p[0] for p in bbox_b), min(p[1] for p in bbox_b)
        bx2, by2 = max(p[0] for p in bbox_b), max(p[1] for p in bbox_b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    except (IndexError, TypeError):
        return 0.0


# ── Translation ──────────────────────────────────────────────────────────────
def translate_texts(ocr_results, src_lang="ja", dst_lang="en"):
    """Translate all unique OCR texts.

    Uses UI_LABEL_DICT for known UI labels, Google Translate for everything else.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        logger.error("deep-translator not installed!")
        return {}

    translator = GoogleTranslator(source=src_lang, target=dst_lang)

    unique_texts = set(r["text"] for r in ocr_results)
    logger.info(f"Translating {len(unique_texts)} unique text segments...")

    translations = {}
    skipped_garbled = 0
    for text in sorted(unique_texts):
        # Check UI label dictionary first (handles garbled OCR of known labels)
        ui_translation = _is_ui_label(text)
        if ui_translation is not None:
            translations[text] = ui_translation
            logger.info(f"  [UI] {text} -> {ui_translation}")
            continue

        # Skip likely garbled OCR results entirely
        # (They'll produce nonsense translations)
        if _is_garbled_translation(text, text):  # check original before translating
            logger.info(f"  [SKIP-GARBLED] {text}")
            skipped_garbled += 1
            continue

        # Google Translate for everything else
        try:
            translated = translator.translate(text)
            if translated:
                # Post-translation garbled check
                if _is_garbled_translation(text, translated):
                    logger.info(f"  [SKIP-GARBLED-TRANS] {text} -> {translated}")
                    skipped_garbled += 1
                    continue
                translations[text] = translated
                logger.info(f"  {text} -> {translated}")
            else:
                translations[text] = text
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {e}")
            translations[text] = text
    
    if skipped_garbled > 0:
        logger.info(f"Skipped {skipped_garbled} garbled translations")

    return translations


# ── Background color sampling ────────────────────────────────────────────────
def sample_bg_color(video_path, timestamp, bbox, padding=4):
    """Sample background color from around a text region in the video."""
    x1 = min(p[0] for p in bbox)
    y1 = min(p[1] for p in bbox)
    x2 = max(p[0] for p in bbox)
    y2 = max(p[1] for p in bbox)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    if not extract_single_frame(video_path, timestamp, tmp_path):
        return "0x1a1a2e"

    try:
        img = Image.open(tmp_path)
        w, h = img.size
        pixels = []

        for dx in range(-padding, 0):
            for y in range(y1, y2, max(1, (y2 - y1) // 5)):
                px = x1 + dx
                if 0 <= px < w and 0 <= y < h:
                    try:
                        p = img.getpixel((px, y))
                        if isinstance(p, tuple) and len(p) >= 3:
                            pixels.append(p[:3])
                    except: pass
                px = x2 - dx - 1
                if 0 <= px < w and 0 <= y < h:
                    try:
                        p = img.getpixel((px, y))
                        if isinstance(p, tuple) and len(p) >= 3:
                            pixels.append(p[:3])
                    except: pass

        for dy in range(-padding, 0):
            for x in range(x1, x2, max(1, (x2 - x1) // 5)):
                py = y1 + dy
                if 0 <= x < w and 0 <= py < h:
                    try:
                        p = img.getpixel((x, py))
                        if isinstance(p, tuple) and len(p) >= 3:
                            pixels.append(p[:3])
                    except: pass
                py = y2 - dy - 1
                if 0 <= x < w and 0 <= py < h:
                    try:
                        p = img.getpixel((x, py))
                        if isinstance(p, tuple) and len(p) >= 3:
                            pixels.append(p[:3])
                    except: pass

        img.close()
        os.unlink(tmp_path)

        if pixels:
            avg_r = sum(p[0] for p in pixels) // len(pixels)
            avg_g = sum(p[1] for p in pixels) // len(pixels)
            avg_b = sum(p[2] for p in pixels) // len(pixels)
            return f"0x{avg_r:02x}{avg_g:02x}{avg_b:02x}"
    except Exception as e:
        logger.debug(f"BG color sampling failed: {e}")
        try: os.unlink(tmp_path)
        except: pass

    return "0x1a1a2e"


# ── Build ffmpeg filter chain ───────────────────────────────────────────────
def build_ffmpeg_filters(ocr_results, translations, video_path, font_path,
                         interval_sec=2.0, canvas_only=False):
    """Build ffmpeg filter chain for inpainting + text overlay."""
    filters = []
    video_info = get_video_info(video_path)
    vid_w = video_info.get("width", 1620)
    vid_h = video_info.get("height", 748)

    for r in ocr_results:
        if canvas_only and r.get("region") != "CANVAS":
            continue

        text = r["text"]
        translated = translations.get(text, None)

        # Skip if not translated (was garbled or non-Japanese)
        if translated is None:
            continue

        # Skip if not Japanese and not translated differently
        if translated == text and not any('\u3040' <= c <= '\u9fff' for c in text):
            continue

        bbox = r["bbox"]
        ts = r["timestamp_sec"]

        x1 = int(min(p[0] for p in bbox))
        y1 = int(min(p[1] for p in bbox))
        x2 = int(max(p[0] for p in bbox))
        y2 = int(max(p[1] for p in bbox))
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        # Time window
        if "end_sec" in r:
            t_start = max(0, r["timestamp_sec"] - interval_sec * 0.3)
            t_end = r["end_sec"] + interval_sec * 0.3
        else:
            t_start = max(0, ts - interval_sec * 0.55)
            t_end = ts + interval_sec * 0.55

        # ── Inpaint: drawbox to cover original text ──
        pad = INPAINT_PADDING
        ix1 = max(0, x1 - pad)
        iy1 = max(0, y1 - pad)
        ix2 = min(vid_w, x2 + pad)
        iy2 = min(vid_h, y2 + pad)
        iw = ix2 - ix1
        ih = iy2 - iy1

        bg_color = sample_bg_color(video_path, ts, bbox, pad)

        drawbox = (
            f"drawbox=x={ix1}:y={iy1}:w={iw}:h={ih}"
            f":color={bg_color}:t=fill"
            f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
        )
        filters.append(drawbox)

        # ── Overlay: drawtext for English translation ──
        font_size = _calc_font_size(translated, bbox_w, bbox_h)
        safe_text = _escape_ffmpeg_text(translated)
        safe_font = font_path.replace("\\", "\\\\\\\\").replace(":", "\\\\:")

        drawtext = (
            f"drawtext=fontfile='{safe_font}'"
            f":text='{safe_text}'"
            f":fontsize={font_size}"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":x={x1}:y={y1}"
            f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
        )
        filters.append(drawtext)

    return filters


def _calc_font_size(text, bbox_w, bbox_h):
    """Calculate font size to fit translated text in the bbox."""
    text_len = len(text)
    if text_len == 0:
        return FONT_SIZE_MIN

    char_width_ratio = 0.5

    font_size_by_width = bbox_w / (text_len * char_width_ratio)
    font_size_by_height = bbox_h * 0.9

    if font_size_by_width < FONT_SIZE_MIN:
        font_size_by_width_2line = bbox_w / (text_len * char_width_ratio / 2)
        font_size_by_height_2line = bbox_h * 0.45
        font_size = min(font_size_by_width_2line, font_size_by_height_2line, FONT_SIZE_MAX)
    else:
        font_size = min(font_size_by_width, font_size_by_height, FONT_SIZE_MAX)

    font_size = max(int(font_size), FONT_SIZE_MIN)
    return font_size


def _escape_ffmpeg_text(text):
    """Escape text for ffmpeg drawtext filter value."""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\\\:")
    text = text.replace("{", "\\\\{")
    text = text.replace("}", "\\\\}")
    text = text.replace("'", "")
    return text


# ── Render with ffmpeg ──────────────────────────────────────────────────────
def render_video(video_path, output_path, filters, video_info):
    """Render translated video using ffmpeg filter chain."""
    if not filters:
        logger.warning("No filters to apply, copying video")
        subprocess.run(["cp", str(video_path), str(output_path)])
        return True

    MAX_FILTERS = 100

    if len(filters) <= MAX_FILTERS:
        return _render_single_pass(video_path, output_path, filters, video_info)
    else:
        drawbox_filters = [f for f in filters if f.startswith("drawbox")]
        drawtext_filters = [f for f in filters if f.startswith("drawtext")]
        logger.info(f"Multi-pass: {len(drawbox_filters)} drawbox + {len(drawtext_filters)} drawtext")

        intermediate = output_path.with_suffix(".inpaint.mp4")
        if not _render_single_pass(video_path, intermediate, drawbox_filters, video_info):
            return False

        result = _render_single_pass(intermediate, output_path, drawtext_filters, video_info)
        intermediate.unlink(missing_ok=True)
        return result


def _render_single_pass(video_path, output_path, filters, video_info):
    """Single ffmpeg pass with given filters."""
    vf = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path)
    ]

    logger.info(f"FFmpeg render: {len(filters)} filters")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr[-2000:]}")
            cmd_no_audio = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-an",
                "-movflags", "+faststart",
                str(output_path)
            ]
            logger.info("Retrying without audio...")
            result = subprocess.run(cmd_no_audio, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed (no audio): {result.stderr[-2000:]}")
                return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out!")
        return False

    if output_path.exists():
        logger.info(f"Rendered: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    return False


# ── Main pipeline ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Video translator: inpaint Japanese + overlay English")
    parser.add_argument("--video", default="/workspace/hiramoto_video.mp4")
    parser.add_argument("--output", default="/workspace/translated_output.mp4")
    parser.add_argument("--interval", type=float, default=2.0, help="OCR frame interval (seconds)")
    parser.add_argument("--backend", default="manga_ocr", choices=["easyocr", "manga_ocr"])
    parser.add_argument("--canvas-only", action="store_true", help="Only translate CANVAS text")
    parser.add_argument("--test", type=float, default=0, metavar="SECONDS",
                        help="Only process first N seconds (0=all)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Install font if needed
    font_path = find_cjk_font()
    if not font_path:
        install_cjk_font()
        font_path = find_cjk_font()
    if not font_path:
        logger.error("No suitable font found!")
        sys.exit(1)
    logger.info(f"Using font: {font_path}")

    # Get video info
    video_info = get_video_info(str(video_path))
    logger.info(f"Video: {video_info}")

    # If test mode, create a clip first
    if args.test > 0:
        test_clip = video_path.with_suffix(".clip.mp4")
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-t", str(args.test), "-c", "copy", str(test_clip)]
        subprocess.run(cmd, capture_output=True, timeout=60)
        video_path = test_clip
        video_info = get_video_info(str(video_path))
        logger.info(f"Test clip: {video_info}")

    # ── Step 1: Extract OCR frames ──
    with tempfile.TemporaryDirectory(prefix="ovt_ocr_") as ocr_dir:
        ocr_frames = extract_ocr_frames(video_path, ocr_dir, args.interval)
        if not ocr_frames:
            logger.error("No frames extracted!")
            sys.exit(1)

        # ── Step 2: Smart dedup ──
        keep_indices = deduplicate_frames(ocr_frames)
        ocr_frames = [ocr_frames[i] for i in keep_indices]
        logger.info(f"After dedup: {len(ocr_frames)} frames to OCR")

        # ── Step 3: OCR with text enlargement ──
        ocr_results = run_ocr_on_frames(
            ocr_frames,
            backend=args.backend,
            video_path=str(video_path),
            orig_width=video_info.get("width"),
            orig_height=video_info.get("height"),
            interval_sec=args.interval,
        )

        # Apply canvas-only filter
        if args.canvas_only:
            ocr_results = [r for r in ocr_results if r.get("region") == "CANVAS"]
            logger.info(f"After canvas filter: {len(ocr_results)} results")

        if not ocr_results:
            logger.warning("No text found! Copying original video.")
            subprocess.run(["cp", str(video_path), str(output_path)])
            sys.exit(0)

        # ── Step 4: Translate (UI dict + Google Translate) ──
        translations = translate_texts(ocr_results)

        # Save OCR + translation data
        data_path = "/workspace/ocr_translations.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({
                "translations": translations,
                "ocr_results": ocr_results,
                "video_info": video_info,
            }, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved OCR data to {data_path}")

        # ── Step 5: Build ffmpeg filters ──
        filters = build_ffmpeg_filters(
            ocr_results, translations, str(video_path),
            font_path, args.interval, args.canvas_only
        )
        logger.info(f"Built {len(filters)} ffmpeg filters")

        for i, f in enumerate(filters[:10]):
            logger.info(f"  Filter {i}: {f[:120]}...")
        if len(filters) > 10:
            logger.info(f"  ... and {len(filters) - 10} more")

        # ── Step 6: Render ──
        success = render_video(video_path, output_path, filters, video_info)

        if success:
            logger.info(f"SUCCESS! Translated video: {output_path}")
        else:
            logger.error("FAILED to render translated video")
            sys.exit(1)

    # Cleanup test clip
    if args.test > 0:
        video_path.with_suffix(".clip.mp4").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
