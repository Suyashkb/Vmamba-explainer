#!/usr/bin/env bash
# render.sh — Render all VMamba explainer scenes and concatenate with ffmpeg.

set -euo pipefail

# ── Change to manim/ directory ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check manim is installed ──────────────────────────────────────────────────
if ! command -v manim &>/dev/null; then
    echo "ERROR: manim is not installed or not in PATH." >&2
    echo "Install it with: pip install manim" >&2
    exit 1
fi

echo "=== Rendering VMamba Explainer scenes ==="

# ── Scene mapping ─────────────────────────────────────────────────────────────
declare -a SCENE_FILES=(
    "scenes/01_motivation.py"
    "scenes/02_ssm_primer.py"
    "scenes/03_selective_scan.py"
    "scenes/04_ss2d.py"
    "scenes/05_vss_block.py"
    "scenes/06_full_vmamba.py"
)

declare -a SCENE_CLASSES=(
    "MotivationScene"
    "SSMPrimerScene"
    "SelectiveScanScene"
    "SS2DScene"
    "VSSBlockScene"
    "FullVMambaScene"
)

declare -a RENDERED_FILES=()

for i in "${!SCENE_FILES[@]}"; do
    SCENE_FILE="${SCENE_FILES[$i]}"
    SCENE_CLASS="${SCENE_CLASSES[$i]}"
    echo ""
    echo "--- Rendering ${SCENE_CLASS} from ${SCENE_FILE} ---"
    manim -qh "${SCENE_FILE}" "${SCENE_CLASS}" --disable_caching
done

echo ""
echo "=== All scenes rendered. Locating output files... ==="

# ── Find rendered mp4 files in manim's media output ──────────────────────────
MEDIA_DIR="media/videos"

for SCENE_CLASS in "${SCENE_CLASSES[@]}"; do
    # manim writes to media/videos/<scene_file_stem>/<quality>/<ClassName>.mp4
    found=$(find "$MEDIA_DIR" -name "${SCENE_CLASS}.mp4" 2>/dev/null | head -1)
    if [[ -z "$found" ]]; then
        echo "WARNING: Could not find rendered file for ${SCENE_CLASS}" >&2
    else
        RENDERED_FILES+=("$found")
        echo "Found: $found"
    fi
done

if [[ ${#RENDERED_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No rendered files found. Cannot concatenate." >&2
    exit 1
fi

# ── Build filelist.txt for ffmpeg ─────────────────────────────────────────────
FILELIST="filelist.txt"
rm -f "$FILELIST"

for f in "${RENDERED_FILES[@]}"; do
    # Use absolute path for safety
    ABS_PATH="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
    echo "file '${ABS_PATH}'" >> "$FILELIST"
done

echo ""
echo "Generated $FILELIST:"
cat "$FILELIST"

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p output

# ── Concatenate with ffmpeg ───────────────────────────────────────────────────
OUTPUT="output/vmamba_explainer.mp4"
echo ""
echo "=== Concatenating with ffmpeg → ${OUTPUT} ==="

ffmpeg -y -f concat -safe 0 -i "$FILELIST" -c copy "$OUTPUT"

echo ""
echo "Done. Output: ${OUTPUT}"
