# Show me what they mean

Freeze one camera image, pair it with a sentence, and have Gemini point at the
object the sentence refers to. The user can then have the explanation spoken by
ElevenLabs.

## Flow

1. Capture one frozen camera image with the current selected-person label.
2. Pair it with a sentence typed or pasted into the manual field.
3. POST the image and sentence to the local `/api/explain` endpoint. The server
   calls Gemini; the key never enters the browser. Requests are bounded and only
   one provider call runs at a time, independently of session controls.
4. Validate the result: one primary object, ambiguous candidates, or no match.
   Reject malformed coordinates and incomplete responses.
5. Display the explanation and normalized boxes on the exact frozen image.
   Ambiguity lets the user choose a candidate; editing the sentence cancels and
   invalidates the previous request and result. The live camera view keeps
   running.
6. Optional: press **Speak with ElevenLabs**. The browser POSTs only the
   explanation text to `/api/speak`; the server calls ElevenLabs and returns a
   WAV that the page plays. Speaking is always an explicit click, never
   automatic, and it never touches the isolated audio path.

## Run and try

Run from the repository root, using the project's Python environment:

```powershell
$env:GEMINI_API_KEY = '<your key>'
$env:ELEVENLABS_API_KEY = '<your key>'
python -m demo.tap_to_select --port 8771
```

Or copy `.env.example` to `.env` (gitignored) and fill in the keys once; the
demo and `python -m demo.speech` read it at startup. Variables already set in
the shell win over the file, and only `ELEVENLABS_`, `GEMINI_` and `ONEVOICE_`
names are read.

Optional overrides:

| Variable | Default |
| --- | --- |
| `ONEVOICE_GEMINI_MODEL` | `gemini-2.5-flash` |
| `ELEVENLABS_VOICE_ID` | a stock ElevenLabs voice |
| `ONEVOICE_ELEVENLABS_MODEL` | `eleven_flash_v2_5` |

In the panel choose **Load CPU demo**, then **Play scripted result**. This works
without any key. **Explain with Gemini** sends the image and sentence to Google,
and **Speak with ElevenLabs** sends the explanation text to ElevenLabs. Missing
keys, quota and provider failures remain errors; there is no silent scripted
fallback. Credentials are environment-only.

To check speech on its own, without the browser:

```powershell
python -m demo.speech "Connect the blue cable to that port."
python -m demo.speech "Connect the blue cable to that port." --save out.pcm
```

For a real snapshot, use an explicitly started camera session, select a visible
person, and freeze the scene. Type or paste their sentence, then press Explain
with Gemini. The snapshot is taken at button time and is not synchronized
automatically to an earlier utterance. No devices open merely from loading this
feature.

## Not done

- Auto-filling the sentence from live captions, preserving track ID and time.
- Buffering timestamped frames so a delayed utterance can pick a matching image.
- Validation with real C270 images and real Gemini and ElevenLabs credentials.
  CPU fixture tests do not establish model accuracy or live isolation.

This is a visual reference suggestion, not a wiring compatibility or safety
check. An ambiguous "that port" should ask a question rather than guess a
connection.

References: https://ai.google.dev/gemini-api/docs/image-understanding and
https://ai.google.dev/gemini-api/docs/structured-output
