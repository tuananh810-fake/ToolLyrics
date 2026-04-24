# ToolLyric

ToolLyric aligns `.mp3` audio to an existing `.lrc` file and exports `lyrics_pro.json` for `LyricsEngine.js`.

## What changed

The current build is optimized for local CPU usage:

- Node.js keeps a persistent Python worker alive instead of spawning Python for every request.
- The worker reuses the loaded Whisper model across jobs.
- CUDA runs use the NVIDIA GPU when PyTorch can see it.
- CPU fallback runs enable Whisper dynamic quantization to reduce inference cost.
- The default model is now `recommended`:
  - `medium` on CPU and CUDA for a better speed/quality balance
  - `large-v3` is still available manually for maximum quality
- The web UI now defaults to the fast CPU path:
  - direct forced alignment from `.lrc` timestamps
  - skip `refine()`
  - skip VAD silence detection
  - skip debug JSON writes
- The slower transcription fallback, refine pass, VAD, and debug output can be enabled from the web UI when a file needs deeper checking.

This avoids long stalls around the 28% direct-alignment step while keeping good output quality for timestamped `.lrc` files.

## Python CLI

The aligner script is:

```powershell
python .\high_precision_word_aligner.py `
  --audio .\song.mp3 `
  --lrc .\song.lrc `
  --output .\lyrics_pro.json `
  --language vi `
  --model recommended `
  --strategy auto `
  --debug-dir .\debug
```

Useful options:

- `--model recommended|medium|large-v3`
- `--strategy auto|direct|transcribe`
- `--skip-refine`
- `--window-pad-ms`
- `--segment-pad-ms`
- `--debug-dir`

## Node local server

Install dependencies:

```powershell
cd D:\ToolLyric
npm install
```

Start the local server:

```powershell
cd D:\ToolLyric
npm start
```

Or:

```powershell
cd D:\ToolLyric
.\run_web_ui.ps1
```

Open:

```text
http://127.0.0.1:5000
```

## Runtime notes

- The bundled Python runtime is used by default:

```text
C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

- `ffmpeg` will fall back to the bundled `imageio-ffmpeg` binary when needed.
- On this machine, PyTorch CUDA is installed for the RTX 4060 Laptop GPU. The web UI `Thiet bi` field can be set to `auto`, `cuda`, or `cpu`.
- Word timestamps are exported in milliseconds with:

```text
Time_ms = round(Time_seconds * 1000)
```

- `stable-ts` does not provide true physical `1ms` alignment accuracy on real audio. The output is millisecond-resolution, not guaranteed 1ms acoustic accuracy.

## Output shape

```json
{
  "song": "Song title",
  "data": [
    {
      "line": "Example lyric line",
      "startTimeMs": 12500,
      "words": [
        { "w": "Example", "s": 12500, "e": 12750 },
        { "w": "lyric", "s": 12750, "e": 13000 }
      ]
    }
  ]
}
```
