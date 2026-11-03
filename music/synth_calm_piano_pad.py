import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
from scipy.io import wavfile
sr=44100
rng=np.random.default_rng(7)
def note(f): return 440*2**((f-69)/12)
def pad(freq,dur,vol=0.18):
    n=int(dur*sr); t=np.arange(n)/sr
    s=np.zeros(n)
    for det in (-0.07,0,0.07):
        f=freq*2**(det/12)
        for h,a in ((1,1),(2,.35),(3,.15),(4,.08)):
            s+=a*np.sin(2*np.pi*f*h*t+rng.uniform(0,6.28))
    att=int(1.4*sr); rel=int(1.8*sr)
    env=np.ones(n); env[:att]=np.linspace(0,1,att)**2; env[-rel:]=np.linspace(1,0,rel)**2
    return s*env*vol/3
def pluck(freq,dur,vol=0.22):
    n=int(dur*sr); t=np.arange(n)/sr
    s=sum(a*np.sin(2*np.pi*freq*h*t)*np.exp(-t*(2.2+h*1.4)) for h,a in ((1,1),(2,.5),(3,.25),(4,.12),(5,.06)))
    env=np.minimum(1,t/0.004)
    return s*env*vol
# Am - F - C - G, 4 beats each at 72 bpm, two rounds + resolve
bpm=72; beat=60/bpm; bar=4*beat
chords=[(57,[57,60,64]),(53,[53,57,60]),(48,[48,52,55]),(55,[55,59,62])]
rounds=3
total=int((rounds*4*bar+4)*sr)
mix=np.zeros(total)
def add(x,start):
    i=int(start*sr); mix[i:i+len(x)]+=x[:max(0,total-i)] if i+len(x)>total else x
for r in range(rounds):
    for ci,(bass,tri) in enumerate(chords):
        t0=(r*4+ci)*bar
        for m in tri: add(pad(note(m+12),bar+1.6),t0)
        add(pad(note(bass-12),bar+1.6,0.22),t0)
        arp=[tri[0]+12,tri[1]+12,tri[2]+12,tri[1]+24] if r>0 else [tri[0]+12]
        if r>0:
            for k in range(8):
                m=[tri[0]+12,tri[1]+12,tri[2]+12,tri[1]+24,tri[2]+12,tri[1]+12,tri[0]+24,tri[1]+12][k]
                add(pluck(note(m+12),1.6),t0+k*beat/2)
        else:
            add(pluck(note(tri[2]+12),2.5,0.18),t0)
# reverb
ir_len=int(3.2*sr); ir=rng.standard_normal(ir_len)*np.exp(-np.arange(ir_len)/(0.9*sr))
b,a=butter(2,5000/(sr/2)); ir=lfilter(b,a,ir)
wet=fftconvolve(mix,ir)[:total]*0.045
out=mix*0.8+wet
out=out/np.max(np.abs(out))*0.8
# fade in/out
fi=int(1.5*sr); fo=int(4*sr)
out[:fi]*=np.linspace(0,1,fi); out[-fo:]*=np.linspace(1,0,fo)
wavfile.write('alphax_calm_piano_pad.wav',sr,(out*32767).astype(np.int16))
print(len(out)/sr)
