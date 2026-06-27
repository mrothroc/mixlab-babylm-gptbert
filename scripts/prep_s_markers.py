import sys, glob, os, numpy as np
from tokenizers import Tokenizer
# Usage: python prep_s_markers.py <TOKENIZER_DIR> <CORPUS_DIR> <OUT_DIR>
#   TOKENIZER_DIR : directory containing tokenizer.json (the reference 16k BabyLM tokenizer)
#   CORPUS_DIR    : directory of *.txt files (the BabyLM Strict-Small corpus)
#   OUT_DIR       : output directory for train_*.bin / val_*.bin shards
if len(sys.argv) != 4:
    sys.exit("usage: prep_s_markers.py <TOKENIZER_DIR> <CORPUS_DIR> <OUT_DIR>")
BASE=sys.argv[1]; CORPUS=sys.argv[2]; OUT=sys.argv[3]
SEG=126          # content tokens per segment (seq_len-2 at base 128)
BOS=1            # <s>
MAGIC=20240520; VERSION=1; HEADER_INTS=256; TOKENS_PER_SHARD=1_000_000; VAL_FRAC=0.1
os.makedirs(OUT, exist_ok=True)
tk=Tokenizer.from_file(os.path.join(BASE, "tokenizer.json"))
# tokenize each line with NO special tokens (we insert <s> ourselves per segment)
ids=[]
for f in sorted(glob.glob(f"{CORPUS}/*.txt")):
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line=line.strip()
            if line: ids.extend(tk.encode(line, add_special_tokens=False).ids)
ids=np.array(ids, dtype=np.uint16)
# insert <s> at the start of every SEG-token segment -> period (SEG+1)=127, ~0.8% density
n_seg=(len(ids)+SEG-1)//SEG
stream=np.empty(len(ids)+n_seg, dtype=np.uint16)
w=0
for off in range(0, len(ids), SEG):
    stream[w]=BOS; w+=1
    chunk=ids[off:off+SEG]; stream[w:w+len(chunk)]=chunk; w+=len(chunk)
stream=stream[:w]
print(f"total tokens {len(stream):,} | <s> count {int((stream==BOS).sum()):,} ({(stream==BOS).mean()*100:.2f}%) | max id {stream.max()}")
# split + write shards (mixlab format)
os.makedirs(OUT, exist_ok=True)
nval=int(len(stream)*VAL_FRAC); val=stream[:nval]; train=stream[nval:]
def write(arr, prefix):
    n=(len(arr)+TOKENS_PER_SHARD-1)//TOKENS_PER_SHARD
    for i in range(n):
        part=arr[i*TOKENS_PER_SHARD:(i+1)*TOKENS_PER_SHARD]
        h=np.zeros(HEADER_INTS, dtype=np.int32); h[0]=MAGIC; h[1]=VERSION; h[2]=len(part)
        with open(f"{OUT}/{prefix}_{i:05d}.bin","wb") as fo:
            fo.write(h.tobytes()); fo.write(part.tobytes())
    return n
nt=write(train,"train"); nv=write(val,"val")
print(f"wrote {nt} train + {nv} val shards to {OUT}")
