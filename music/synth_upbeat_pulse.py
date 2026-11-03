import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
from scipy.io import wavfile
sr=44100; rng=np.random.default_rng(3)
bpm=124; beat=60/bpm; bar=4*beat; bars=24
total=int((bars*bar+3)*sr); mix=np.zeros(total)
def hz(m): return 440*2**((m-69)/12)
def put(x,t,buf):
    i=int(t*sr); e=min(len(buf),i+len(x))
    if i<len(buf): buf[i:e]+=x[:e-i]
def lp(x,fc,o=2): b,a=butter(o,fc/(sr/2)); return lfilter(b,a,x)
def hp(x,fc,o=2): b,a=butter(o,fc/(sr/2),'high'); return lfilter(b,a,x)
def kick():
    n=int(0.32*sr); t=np.arange(n)/sr
    f=45+110*np.exp(-t*28); ph=2*np.pi*np.cumsum(f)/sr
    return np.sin(ph)*np.exp(-t*9)*1.0 + 0.3*np.sin(ph*2)*np.exp(-t*30)
def clap():
    n=int(0.2*sr); t=np.arange(n)/sr
    x=hp(rng.standard_normal(n),1200)*np.exp(-t*22)
    return lp(x,7000)*0.5
def hat(open_=False):
    n=int((0.18 if open_ else 0.05)*sr); t=np.arange(n)/sr
    return hp(rng.standard_normal(n),7000)*np.exp(-t*(18 if open_ else 80))*0.22
def saw(f,n):
    t=np.arange(n)/sr; return 2*((f*t)%1)-1
def chord(notes,dur):
    n=int(dur*sr); s=np.zeros(n)
    for m in notes:
        for d in (-0.12,0.12): s+=saw(hz(m)*2**(d/12),n)
    s=lp(s,3200,2)/ (len(notes)*2)
    return s*0.6
def bassn(m,dur):
    n=int(dur*sr); t=np.arange(n)/sr
    s=np.sin(2*np.pi*hz(m)*t)+0.35*np.sin(4*np.pi*hz(m)*t)
    return s*np.minimum(1,t/0.01)*np.exp(-t*3)*0.9
def pluck(m,dur=0.5):
    n=int(dur*sr); t=np.arange(n)/sr
    s=saw(hz(m),n)*np.exp(-t*7)
    return lp(s,5000)*0.28
prog=[(48,[60,64,67]),(55,[59,62,67]),(57,[60,64,69]),(53,[60,65,69])]
kb=np.zeros(total); chords_b=np.zeros(total); bass_b=np.zeros(total); perc=np.zeros(total); lead=np.zeros(total)
k=kick(); c=clap(); h=hat(); ho=hat(True)
for b in range(bars):
    t0=b*bar; root,tri=prog[b%4]
    full = b>=4 and not (b in (11,12))   # breakdown at bars 11-12
    # chords always (softer in intro)
    put(chord(tri,bar+0.3)*(0.6 if b<4 else 1.0),t0,chords_b)
    # hats
    for q in range(8):
        put(ho if q%2==1 else h, t0+q*beat/2, perc)
    if b>=2:
        for q in range(16): put(h*0.5,t0+q*beat/4,perc)
    if full:
        for q in range(4):
            put(k,t0+q*beat,kb)
        for q in (1,3): put(c,t0+q*beat,perc)
        for q in range(8):
            put(bassn(root-12 if q%2==0 else root,beat/2*0.95),t0+q*beat/2,bass_b)
        for q,m in enumerate([tri[0]+12,tri[2]+12,tri[1]+12,tri[2]+12,tri[0]+24,tri[2]+12,tri[1]+12,tri[0]+12]):
            put(pluck(m),t0+q*beat/2,lead)
    elif b>=4:  # breakdown: riser claps
        for q in range(8): put(c*0.5,t0+q*beat/2 if b==12 else t0+q*beat,perc)
# sidechain pump on chords+bass+lead
env=np.ones(total)
for b in range(bars):
    if b>=4 and b not in (11,12):
        for q in range(4):
            i=int((b*bar+q*beat)*sr); n=int(beat*0.75*sr)
            curve=1-np.exp(-np.arange(n)/(0.12*sr))*0.85
            env[i:i+n]=np.minimum(env[i:i+n],curve[:min(n,total-i)])
mixd=kb*1.0+(chords_b*0.55+bass_b*0.8+lead)*env+perc*0.9
ir_len=int(1.6*sr); ir=rng.standard_normal(ir_len)*np.exp(-np.arange(ir_len)/(0.35*sr)); ir=lp(ir,6000)
wet=fftconvolve(chords_b*env+lead*env,ir)[:total]*0.03
out=mixd+wet
out=np.tanh(out*1.3)
out=out/np.max(np.abs(out))*0.85
fi=int(0.8*sr); fo=int(3*sr)
out[:fi]*=np.linspace(0,1,fi); out[-fo:]*=np.linspace(1,0,fo)
wavfile.write('alphax_upbeat_pulse.wav',sr,(out*32767).astype(np.int16))
print(len(out)/sr)
