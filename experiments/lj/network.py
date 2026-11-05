"""Spatial score correction in physical coordinates with Gaussian preconditioning."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from preprocessing import gaussian_parameters
from common import resolve_recorded_path
ROOT=Path(__file__).resolve().parent


class ConvBlock(nn.Module):
    def __init__(self,ch,embedding):
        super().__init__()
        self.a=nn.Conv2d(ch,ch,3,padding=1)
        self.b=nn.Conv2d(ch,ch,3,padding=1)
        self.cond=nn.Linear(embedding,2*ch)
        nn.init.zeros_(self.cond.weight);nn.init.zeros_(self.cond.bias)
    def forward(self,x,c):
        scale,shift=self.cond(c).chunk(2,dim=-1)
        h=self.a(torch.nn.functional.silu(x))
        h=h*(1+.1*scale[:,:,None,None])+.1*shift[:,:,None,None]
        return x+.25*self.b(torch.nn.functional.silu(h))


class PhysicalScoreModel(nn.Module):
    def __init__(self,case_path,dimension=None,width=48,embedding=128,processed_directory=None,temperature_grid=None,grid_size=18):
        super().__init__()
        case_path=resolve_recorded_path(case_path)
        processed_directory=resolve_recorded_path(processed_directory or case_path/'processed')
        qref=np.load(case_path/'geometry.npz')['qref']
        self.atoms=len(qref)
        dimension=2*self.atoms
        self.dimension=dimension
        self.config=dict(architecture='physical_unet',case_path=case_path.relative_to(ROOT.parents[1]).as_posix(),dimension=dimension,width=width,embedding=embedding,processed_directory=processed_directory.relative_to(ROOT.parents[1]).as_posix())
        if temperature_grid is not None:
            self.config['temperature_grid']=list(map(float,temperature_grid))
        self.grid_size=int(grid_size);self.grid_cells=self.grid_size**2
        if grid_size!=18:self.config['grid_size']=self.grid_size
        self.parameters_file=processed_directory/'gaussian_parameters.npz'
        parameters=np.load(self.parameters_file)
        indices=[]
        for x,y in qref:
            j=int(round(y/(np.sqrt(3)/2*1.1161533029896453)))
            i=int(round(x/1.1161533029896453-.5*(j%2)))
            indices.append((j+self.grid_size//2)*self.grid_size+i+self.grid_size//2)
        self.register_buffer('indices',torch.tensor(indices,dtype=torch.int64))
        self.register_buffer('reference',torch.tensor(qref.ravel(),dtype=torch.float32))
        grid=np.zeros((4,self.grid_size,self.grid_size),dtype=np.float32)
        grid[0].ravel()[indices]=qref[:,0]/10
        grid[1].ravel()[indices]=qref[:,1]/10
        grid[2].ravel()[indices]=1
        grid[3]=np.broadcast_to((np.arange(self.grid_size)%2)[:,None],(self.grid_size,self.grid_size))
        self.register_buffer('grid_features',torch.tensor(grid))
        ts=np.arange(2,11)/100 if temperature_grid is None else np.asarray(temperature_grid,dtype=np.float64)
        if ts.ndim!=1 or not len(ts) or np.any(ts<=0) or np.any(np.diff(ts)<=0):
            raise ValueError('Temperature grid must be positive, nonempty and strictly increasing.')
        means=[];chols=[];scales=[]
        for t in ts:
            mu,cov=gaussian_parameters(t,parameters)
            means.append(mu);chols.append(np.linalg.cholesky(cov));scales.append(np.sqrt(np.trace(cov)/dimension))
        self.register_buffer('temperatures',torch.tensor(ts,dtype=torch.float32))
        self.register_buffer('means',torch.tensor(np.array(means),dtype=torch.float32))
        self.register_buffer('cholesky',torch.tensor(np.array(chols),dtype=torch.float32))
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
        self.register_buffer('frequencies',torch.arange(1,17,dtype=torch.float32)*math.pi)
        self.embed=nn.Sequential(nn.Linear(36,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        self.input=nn.Conv2d(6,width,3,padding=1)
        self.level0=nn.ModuleList([ConvBlock(width,embedding) for _ in range(3)])
        self.down=nn.Conv2d(width,2*width,3,stride=2,padding=1)
        self.level1=nn.ModuleList([ConvBlock(2*width,embedding) for _ in range(3)])
        self.down2=nn.Conv2d(2*width,4*width,3,stride=2,padding=1)
        self.bottleneck=nn.ModuleList([ConvBlock(4*width,embedding) for _ in range(3)])
        self.up1=nn.Conv2d(6*width,2*width,3,padding=1)
        self.up0=nn.Conv2d(3*width,width,3,padding=1)
        self.output_blocks=nn.ModuleList([ConvBlock(width,embedding) for _ in range(2)])
        self.output=nn.Conv2d(width,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.sampling_index=None
        self.sampling_active=False
        self.register_buffer('sampling_cholesky',torch.zeros(dimension,dimension),persistent=False)
        self.register_buffer('sampling_mean',torch.zeros(dimension),persistent=False)
        self.register_buffer('sampling_scale',torch.ones(()),persistent=False)
    @torch.no_grad()
    def configure_temperature(self,temp):
        if not float(self.temperatures.min())-1e-7<=temp<=float(self.temperatures.max())+1e-7:
            raise ValueError('Requested temperature lies outside this model temperature grid.')
        if float(torch.min(abs(self.temperatures-temp)).item())<1e-7:
            k=int(torch.argmin(abs(self.temperatures-temp)).item())
        else:
            parameters=np.load(self.parameters_file)
            mean,cov=gaussian_parameters(temp,parameters)
            self.temperatures[0]=temp
            self.means[0].copy_(torch.as_tensor(mean,dtype=self.means.dtype,device=self.means.device))
            self.cholesky[0].copy_(torch.as_tensor(np.linalg.cholesky(cov),dtype=self.cholesky.dtype,device=self.cholesky.device))
            self.scales[0]=np.sqrt(np.trace(cov)/self.dimension)
            k=0
        self.sampling_index=k;self.sampling_active=True
        self.sampling_cholesky.copy_(self.cholesky[k])
        self.sampling_mean.copy_(self.means[k]);self.sampling_scale.copy_(self.scales[k])
    def forward(self,z,time,temperature):
        if self.sampling_active:
            L=self.sampling_cholesky;scale=self.sampling_scale
            with torch.autocast('cuda',enabled=False):
                physical=z.float()@L.float().T
            displacement=(physical+self.sampling_mean-self.reference)/scale
            groups=None
        else:
            ti=torch.argmin(abs(temperature[:,None]-self.temperatures[None,:]),dim=1)
            groups=[(int(k),torch.nonzero(ti==k).flatten()) for k in torch.unique(ti).tolist()]
            physical=torch.empty_like(z)
            for k,ids in groups:
                physical=physical.index_copy(0,ids,z.index_select(0,ids)@self.cholesky[k].T)
            scale=self.scales[ti,None]
            displacement=(physical+self.means[ti]-self.reference)/scale
        grid=torch.zeros((len(z),2,self.grid_cells),device=z.device,dtype=z.dtype)
        grid[:,:,self.indices]=displacement.reshape(-1,self.atoms,2).transpose(1,2)
        grid=grid.reshape(-1,2,self.grid_size,self.grid_size)
        inputs=torch.cat([grid,self.grid_features[None].expand(len(z),-1,-1,-1)],dim=1)
        phi=time[:,None]*self.frequencies[None]
        ratio=temperature/.05
        c=self.embed(torch.cat([phi.sin(),phi.cos(),time[:,None],ratio.log()[:,None],
                               (ratio-1)[:,None],ratio.sqrt()[:,None]],dim=1))
        x=self.input(inputs)
        for block in self.level0:x=block(x,c)
        y=self.down(x)
        for block in self.level1:y=block(y,c)
        b=self.down2(y)
        for block in self.bottleneck:b=block(b,c)
        b=torch.nn.functional.interpolate(b,size=y.shape[-2:],mode='bilinear',align_corners=False)
        y=self.up1(torch.cat([y,b],dim=1))
        y=torch.nn.functional.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=False)
        x=self.up0(torch.cat([x,y],dim=1))
        for block in self.output_blocks:x=block(x,c)
        h=self.output(torch.nn.functional.silu(x)).reshape(-1,2,self.grid_cells)[:,:,self.indices]
        h=h.transpose(1,2).reshape(-1,self.dimension).float()/scale
        if groups is None:
            with torch.autocast('cuda',enabled=False):
                return h.float()@L.float()
        result=torch.empty_like(z)
        for k,ids in groups:
            result=result.index_copy(0,ids,h.index_select(0,ids)@self.cholesky[k])
        return result


def build_model(config):
    config=dict(config)
    config.pop('architecture',None)
    return PhysicalScoreModel(**config)
