import numpy as np, time
from PIL import Image
import tt_render as R
t=time.time()
buf=R.render_buffers(1800,1686,az=45,el=21,dist=225,fov=24,target=(0,16.5,0),light_pos=(90,300,240),fill_dir=(0.5,0.3,-0.8))
print('buf',time.time()-t)
def save(body,acc,path):
    img=R.composite(buf,body,acc,key=0.78,ambk=0.52)
    Image.fromarray(img).resize((900,843),Image.LANCZOS).save(path,quality=87,optimize=True,progressive=True)
D='img/'
save('#7E2428','#E8DFD0',D+'tt-burgundy-cream.jpg')
save('#9CAF94','#E8DFD0',D+'tt-sage-cream.jpg')
C={'cream':'#E8DFD0','sunset':'#E8753D','brick':'#8E3F3B','seaside':'#8FD9CF','blue':'#315C9A','white':'#F7F6F2','graphite':'#252525'}
for b in C:
    for a in C: save(C[b],C[a],D+f'c-{b}-{a}.jpg')
print('done',time.time()-t)
