# ToolLyrics local alignment boost patch

This patch is intended for local-only ToolLyrics usage. It prioritizes successful MP3 + LRC alignment over maximum speed.

## Apply

Copy `apply_toollyrics_local_alignment_boost.py` into the ToolLyrics repository root, then run:

```powershell
python .\apply_toollyrics_local_alignment_boost.py
```

or:

```bash
python apply_toollyrics_local_alignment_boost.py
```

The script creates backups next to each edited file:

- `high_precision_word_aligner.py.bak.local-boost`
- `python_worker.py.bak.local-boost`
- `server.js.bak.local-boost`

Restart the Node server after applying the patch.

## What it changes

- Makes LRC reading more tolerant: UTF-8 BOM, UTF-8, CP1258, CP1252, Latin-1 fallback.
- Repairs common mojibake text before token matching.
- Cleans inline karaoke tags like `<00:12.34>`, zero-width chars, music-note symbols, speaker labels, and visual separators.
- Uses wider local windows for forced alignment, useful when LRC line timestamps drift.
- Lowers the fuzzy token threshold from `0.72` to `0.66` to recover more Vietnamese/diacritic/noisy ASR matches.
- Retries failed stable-ts forced alignment without VAD and then with shorter minimum word duration.
- Switches the local web flow to `auto` strategy by default, so it can fall back from direct LRC alignment to transcription-assisted matching.
- Sends non-JSON library output to stderr so the Node/Python JSON protocol is less likely to break.
- Adds robust default parameters for local quality:
  - `windowPadMs: 650`
  - `segmentPadMs: 350`
  - `tailPadMs: 8000`
  - `minWordDur: 0.03`
  - `refinePrecision: 0.03-0.04`

## Revert

Stop the server, then restore backups:

```powershell
copy .\high_precision_word_aligner.py.bak.local-boost .\high_precision_word_aligner.py
copy .\python_worker.py.bak.local-boost .\python_worker.py
copy .\server.js.bak.local-boost .\server.js
```
