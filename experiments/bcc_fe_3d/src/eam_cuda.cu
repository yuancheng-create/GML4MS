// Single-species EAM, FP64 throughout. Distances use the upper-triangular deformation F.
// Neighbor lists are directed; forces are accumulated per atom, without atomics.
extern "C" {
__device__ void poly(double x, double h, const double* c, int n, double &v, double &d) {
  double t=x/h; int k=max(0,min((int)t,n-1)); double u=fmax(0.,fmin(1.,t-k));
  c+=4*k; v=c[0]+u*(c[1]+u*(c[2]+u*c[3]));
  d=(c[1]+u*(2*c[2]+3*u*c[3]))/h;
}
__device__ void delta(const double* q,int i,int j,double L,double stretch,double shear,
                     double &x,double &y,double &z) {
  y=q[3*i+1]-q[3*j+1]; z=q[3*i+2]-q[3*j+2];
  x=(q[3*i]-q[3*j]-shear*y)/stretch;
  x-=L*nearbyint(x/L); y-=L*nearbyint(y/L); z-=L*nearbyint(z/L);
  x=stretch*x+shear*y;
}
__global__ void cells(const double* q,int n,int nc,double L,double stretch,double shear,
                      int cap,int* count,int* atoms,int* which,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  double y=q[3*i+1]/L,z=q[3*i+2]/L,x=(q[3*i]-shear*q[3*i+1])/(stretch*L);
  int ix=min(nc-1,(int)((x-floor(x))*nc));
  int iy=min(nc-1,(int)((y-floor(y))*nc));
  int iz=min(nc-1,(int)((z-floor(z))*nc));
  int c=(ix*nc+iy)*nc+iz;which[i]=c;int slot=atomicAdd(count+c,1);
  if(slot>=cap)atomicOr(error,1);else atoms[c*cap+slot]=i;
}
__global__ void neighbors(const double* q,int n,int nc,double L,double stretch,double shear,
                          double radius2,int cap,const int* count,const int* atoms,
                          const int* which,int maxnb,int* numbers,int* list,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int c=which[i],cx=c/(nc*nc),cy=(c/nc)%nc,cz=c%nc,k=0;
  int start=nc<3?0:-1,end=nc<3?nc:2;
  for(int dx=start;dx<end;dx++)for(int dy=start;dy<end;dy++)for(int dz=start;dz<end;dz++) {
    int xx=(cx+dx+nc)%nc,yy=(cy+dy+nc)%nc,zz=(cz+dz+nc)%nc;
    int cc=(xx*nc+yy)*nc+zz;
    for(int h=0;h<min(cap,count[cc]);h++) {
      int j=atoms[cc*cap+h];if(i==j)continue;double x,y,z;
      delta(q,i,j,L,stretch,shear,x,y,z);
      if(x*x+y*y+z*z<radius2) {
        if(k<maxnb)list[k*n+i]=j;else atomicOr(error,1);k++;
      }
    }
  }
  numbers[i]=min(k,maxnb);
}
__global__ void density(const double* q,int n,const int* numbers,const int* list,
                        double L,double stretch,double shear,double rc2,
                        const double* den,const double* emb,int nr,int nrho,
                        double dr,double drho,double* rho,double* fp,double* energy,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double sum=0.;
  for(int k=0;k<numbers[i];k++) {
    int j=list[k*n+i];double x,y,z;delta(q,i,j,L,stretch,shear,x,y,z);
    double rs=x*x+y*y+z*z;if(rs<rc2){double val,d;poly(sqrt(rs),dr,den,nr,val,d);sum+=val;}
  }
  rho[i]=sum;if(sum>nrho*drho||sum<0)atomicOr(error,4);
  double val,d;poly(sum,drho,emb,nrho,val,d);fp[i]=d;energy[i]=val;
}
__global__ void force(const double* q,int n,const int* numbers,const int* list,
                      double L,double stretch,double shear,double rc2,
                      const double* den,const double* rp,int nr,double dr,
                      const double* fp,double* f,double* energy) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double fx=0,fy=0,fz=0,pe=0;
  for(int k=0;k<numbers[i];k++) {
    int j=list[k*n+i];double x,y,z;delta(q,i,j,L,stretch,shear,x,y,z);
    double rs=x*x+y*y+z*z;
    if(rs<rc2){double r=sqrt(rs),val,dw,zz,dz;poly(r,dr,den,nr,val,dw);poly(r,dr,rp,nr,zz,dz);
      double phi=zz/r,scale=-((fp[i]+fp[j])*dw+(dz-phi)/r)/r;
      fx+=scale*x;fy+=scale*y;fz+=scale*z;pe+=.5*phi;
    }
  }
  f[3*i]=fx;f[3*i+1]=fy;f[3*i+2]=fz;energy[i]+=pe;
}
__device__ unsigned long long mix64(unsigned long long x) {
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
__device__ double normal(unsigned long long seed,unsigned long long counter) {
  unsigned long long base=mix64(seed);
  double u=((mix64(base+2*counter)>>11)+.5)*1.11022302462515654042e-16;
  double v=((mix64(base+2*counter+1)>>11)+.5)*1.11022302462515654042e-16;
  return sqrt(-2*log(u))*cos(6.2831853071795864769*v);
}
__global__ void drift(double* q,double* v,const double* f,const double* built,int n,
                      double dt,double acc,double decay,double noise,double skin2,
                      unsigned long long seed,const unsigned long long* step,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double displacement=0;
  for(int d=0;d<3;d++) {
    int k=3*i+d;double vel=v[k]+.5*dt*acc*f[k];
    q[k]+=.5*dt*vel;
    double eta=normal(seed,(*step)*(3ULL*n)+(unsigned long long)k);
    vel=decay*vel+noise*eta;
    q[k]+=.5*dt*vel;v[k]=vel;double dx=q[k]-built[k];displacement+=dx*dx;
  }
  if(displacement>skin2)atomicOr(error,2);
}
__global__ void kick(double* v,const double* f,int n,double dt,double acc,unsigned long long* step) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<3*n)v[k]+=.5*dt*acc*f[k];
  if(k==0)(*step)++;
}
}
