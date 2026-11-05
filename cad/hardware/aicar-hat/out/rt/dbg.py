import sys,math
sys.path.insert(0,'.')
import pcbnew,wx
app=wx.App(False)
import pcb_tool as T
b=pcbnew.LoadBoard('out/rt/r.kicad_pcb')
to=pcbnew.ToMM;MM=pcbnew.FromMM
fits,place,tracks,pads=T.stitch(b,targets=True)
for name,(px,py) in {'R15.2':(115.5,138.32),'C2.2':(136.28,132.0),'C14.2':(112.0,118.47)}.items():
    near=[(n,round(ax-100,1),round(ay-100,1),round(bx-100,1),round(by-100,1),round(2*hw,2)) for ax,ay,bx,by,hw,n in tracks if n!='GND' and T.seg_dist(px,py,ax,ay,bx,by)<2.2]
    print(name,'tracks',near[:9])
    ok=[]
    for dx in [i*0.2 for i in range(-18,19)]:
        for dy in [i*0.2 for i in range(-18,19)]:
            if fits(px+dx,py+dy): ok.append((round(px+dx-100,1),round(py+dy-100,1)))
    print('  fit spots',len(ok),ok[:14])
