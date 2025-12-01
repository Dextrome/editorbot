import soundfile as sf, numpy as np
fn ='output/new_heal_250_thresh_0p01.wav'
arr,sr=sf.read(fn)
mono=np.mean(np.abs(arr),axis=1)
print('shape',arr.shape)
thr=0.01
small=mono<thr
runs=[]
start=None
for i,v in enumerate(small):
    if v and start is None:
        start=i
    elif not v and start is not None:
        runs.append((start,i-start)); start=None
if start is not None:
    runs.append((start,len(small)-start))
large=[r for r in runs if r[1]>=int(0.1*sr)]
print('Large runs:',len(large))
if large:
    for s,l in large[:10]:
        print('run',s,l,'ms',l/sr*1000)
    s,l=large[0]
    print('Segment min/max before heal', np.min(arr[s-10:s+10]), np.max(arr[s-10:s+10]))
    print('Segment min/max after heal region', np.min(arr[s:s+l]), np.max(arr[s:s+l]))
print('Max run len', max(runs, key=lambda x:x[1])[1] if runs else 0)
