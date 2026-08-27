import json, subprocess, tempfile
from pathlib import Path
from transcribe import load_api_key, call_scribe
SRC=r"C:/Users/Jaime Herrera/Videos/youtube/36 (01-06-26)/2026-06-01 21-51-41.mp4"
base=Path(r"C:/dev/Video editing/projects/2026-06-01_video-36/edit")
S,E=718.0,746.0
key=load_api_key()
with tempfile.TemporaryDirectory() as tmp:
    wav=Path(tmp)/"cta.wav"
    subprocess.run(["ffmpeg","-y","-ss",str(S),"-to",str(E),"-i",SRC,"-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    resp=call_scribe(wav,key,language="es")
new=[{"text":w["text"],"start":round(w["start"]+S,3),"end":round(w["end"]+S,3)} for w in resp["words"] if w["type"]=="word" and w["text"].strip()]
print("recovered",len(new),"words:")
for w in new: print(f"  {w['start']:.2f}-{w['end']:.2f} {w['text']!r}")
# merge into words_corrected: drop existing in [S,E], add new
p=base/"transcripts"/"words_corrected.json"
W=json.load(open(p,encoding="utf-8"))
W=[w for w in W if not (S<=w["start"]<E)]
W.extend(new)
W.sort(key=lambda w:w["start"])
json.dump(W,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=0)
print("total now",len(W))
