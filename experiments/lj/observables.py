"""Common stress/RDF operators for MD and generated nonperiodic patches."""
import numpy as np
from numba import njit
from md_core import RC,DUC,bond_fraction

A=1.1161533029896453
HX=4*A
HY=2*np.sqrt(3)*A
CORE_AREA=4*HX*HY
DISK_RADIUS=1.5*A
DISK_AREA=np.pi*DISK_RADIUS**2
DR=.005*A
NBINS=680
RMAX=DR*NBINS
RHO0=2/(np.sqrt(3)*A*A)
STAT_NAMES=['rect_sxx','rect_syy','rect_sxy','rect_stress',
            'disk_sxx','disk_syy','disk_sxy','disk_stress',
            'core_number','disk_number','core_ux','core_uy',
            'min_pair_distance','max_displacement','near_mean_r','near_width',
            'kinetic_temperature','energy_per_atom']


@njit(cache=True)
def rect_fraction(mx,my,rx,ry):
    lo,hi=-.5,.5
    for m,r,h in [(mx,rx,HX),(my,ry,HY)]:
        if abs(r)<1e-14:
            if abs(m)>h:
                return 0.
        else:
            x,y=(-h-m)/r,(h-m)/r
            lo=max(lo,min(x,y));hi=min(hi,max(x,y))
    return max(0.,hi-lo)


def candidates(qref,margin=.65):
    d=qref[:,None]-qref[None,:]
    r=np.linalg.norm(d,axis=-1)
    i,j=np.where(np.triu(r<RMAX+2*margin,1))
    return np.column_stack([i,j]).astype(np.int64)


@njit(cache=True)
def measure(q,qref,temp,pairs,selected,core_tags,periodic_box,periodic):
    n=len(q)
    pos=q.copy()
    if periodic:
        pos-=periodic_box*np.rint(pos/periodic_box)
    in_core=np.empty(n,dtype=np.bool_)
    nc,nd=0,0
    ux,uy,maxdisp=0.,0.,0.
    for i in range(n):
        in_core[i]=(-HX<=pos[i,0]<HX and -HY<=pos[i,1]<HY)
        nc+=int(in_core[i])
        nd+=int(pos[i,0]**2+pos[i,1]**2<DISK_RADIUS**2)
        maxdisp=max(maxdisp,np.sqrt(np.sum((q[i]-qref[i])**2)))
        if core_tags[i]:
            ux+=q[i,0]-qref[i,0];uy+=q[i,1]-qref[i,1]
    ux/=np.sum(core_tags);uy/=np.sum(core_tags)
    rect=np.zeros(3);disk=np.zeros(3)
    hist=np.zeros((3,NBINS),dtype=np.int64)
    rmin=1e30
    sumr,sumr2,number=0.,0.,0
    # For the n=8 baseline C8 is its entire periodic cell: all periodic
    # pieces of each bond contribute, so the total rectangle fraction is 1.
    entire_cell=periodic and abs(periodic_box[0]-2*HX)<1e-8
    for k in range(len(pairs)):
        i,j=pairs[k,0],pairs[k,1]
        rx,ry=q[i,0]-q[j,0],q[i,1]-q[j,1]
        if periodic:
            rx-=periodic_box[0]*np.rint(rx/periodic_box[0])
            ry-=periodic_box[1]*np.rint(ry/periodic_box[1])
        r2=rx*rx+ry*ry
        r=np.sqrt(r2)
        rmin=min(rmin,r)
        if r<RMAX:
            b=int(r/DR)
            hist[0,b]+=int(in_core[i])+int(in_core[j])
            hist[1,b]+=int(selected[i])+int(selected[j])
            if .8*A<=r<1.3*A:
                w=int(selected[i])+int(selected[j])
                sumr+=w*r;sumr2+=w*r2;number+=w
        if r<RC:
            mx,my=q[i,0]-.5*rx,q[i,1]-.5*ry
            if periodic:
                mx-=periodic_box[0]*np.rint(mx/periodic_box[0])
                my-=periodic_box[1]*np.rint(my/periodic_box[1])
            b=1. if entire_cell else rect_fraction(mx,my,rx,ry)
            bd=bond_fraction(mx,my,rx,ry,DISK_RADIUS)
            r6=r2**-3
            pref=24*(2*r6*r6-r6)/r2+DUC/r
            rect[0]-=pref*rx*rx*b/CORE_AREA
            rect[1]-=pref*ry*ry*b/CORE_AREA
            rect[2]-=pref*rx*ry*b/CORE_AREA
            disk[0]-=pref*rx*rx*bd/DISK_AREA
            disk[1]-=pref*ry*ry*bd/DISK_AREA
            disk[2]-=pref*rx*ry*bd/DISK_AREA
    for i in range(n):
        r=np.sqrt(pos[i,0]**2+pos[i,1]**2)
        if r<RMAX:
            hist[2,int(r/DR)]+=1
    rect[:2]-=temp*nc/CORE_AREA
    disk[:2]-=temp*nd/DISK_AREA
    stats=np.empty(18)
    stats[:3]=rect;stats[3]=.5*(rect[0]+rect[1])
    stats[4:7]=disk;stats[7]=.5*(disk[0]+disk[1])
    stats[8]=nc;stats[9]=nd;stats[10]=ux;stats[11]=uy
    stats[12]=rmin;stats[13]=maxdisp
    stats[14]=sumr/number
    stats[15]=np.sqrt(max(0.,sumr2/number-stats[14]**2))
    stats[16:]=np.nan
    return stats,hist


@njit(cache=True)
def measure_batch(coords,qref,temp,pairs,selected,core_tags):
    stats=np.empty((len(coords),18))
    blockhist=np.zeros(((len(coords)+199)//200,3,NBINS),dtype=np.int64)
    for j in range(len(coords)):
        dmax=np.max(np.sqrt(np.sum((coords[j]-qref)**2,axis=1)))
        use=pairs
        if dmax>.65:
            use=np.empty((len(qref)*(len(qref)-1)//2,2),dtype=np.int64)
            k=0
            for a in range(len(qref)):
                for b in range(a+1,len(qref)):
                    use[k,0]=a;use[k,1]=b;k+=1
        s,h=measure(coords[j],qref,temp,use,selected,core_tags,np.zeros(2),False)
        stats[j]=s
        blockhist[j//200]+=h
    return stats,blockhist
