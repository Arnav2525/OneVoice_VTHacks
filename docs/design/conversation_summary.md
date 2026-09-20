# Conversation summary

Live captions of the selected person are transcribed on this computer. When the
user presses **Summarize with Gemini**, the caption text collected in this
session is sent to Gemini and a short summary comes back. The user can then have
the summary read aloud by ElevenLabs.

## Flow

1. Captions run locally (faster-whisper) on the isolated audio of the selected
   person. Each finished caption line is also kept in a bounded, in-memory list
   (the newest 300 lines) labelled with the person's name.
2. **Summarize with Gemini** POSTs an empty body to `/api/summarize`. The server
   reads its own transcript, so the browser never sends caption text and cannot
   choose what is sent. The server keeps the key, allows one request at a time,
   and caps the transcript at 12,000 characters (newest lines win).
3. Gemini returns a summary and up to five key points as JSON. The server
   validates the shape and length before showing it.
4. Optional: **Speak with ElevenLabs** POSTs only the summary text to
   `/api/speak` and plays the returned WAV. Speaking happens on an explicit
   click, or automatically only if the user turns on the opt-in checkbox, which
   is off by default and resets on reload.

The transcript lives in memory only. It is cleared when a new session starts and
when the app closes, and it is never written to disk.

## Run and try

```powershell
$env:GEMINI_API_KEY = '<your key>'
$env:ELEVENLABS_API_KEY = '<your key>'
python -m demo.tap_to_select --live --port 8771
```

Or copy `.env.example` to `.env` (gitignored) and fill in the keys once; the
demo reads it at startup. Variables already set in the shell win over the file,
and only `ELEVENLABS_`, `GEMINI_` and `ONEVOICE_` names are read.

| Variable | Default |
| --- | --- |
| `ONEVOICE_GEMINI_MODEL` | `gemini-2.5-flash` |
| `ELEVENLABS_VOICE_ID` | a stock ElevenLabs voice |
| `ONEVOICE_ELEVENLABS_MODEL` | `eleven_flash_v2_5` |

**Play scripted demo** shows a canned summary with no keys and no captions. To
check speech on its own, run `python -m demo.speech "Testing one two three."`.

## What leaves the computer

- Summarize: the selected person's caption text, to Google. No audio or video.
- Speak: the summary text only, to ElevenLabs.

Both are off until pressed. This is a change from "nothing leaves the computer",
so the privacy policy sections on transcripts and summaries must say so.

## Not done

- Captions have not been run end to end; they need Dolphin on a GPU machine.
- Gemini and ElevenLabs have not been called with real keys.
- No transcript view in the page; the user cannot yet see the text before it is
  sent.
- Summaries are not saved; they disappear on reload.
