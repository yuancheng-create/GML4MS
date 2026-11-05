"""Heun probability-flow sampling for the V2 physical score networks."""
import math
import torch


class CapturedHeun:
    """Replay the same Heun grid while avoiding Python/kernel launch overhead."""
    def __init__(self,model,temperature,steps,batch,amp):
        device=next(model.parameters()).device
        self.x=torch.zeros(batch,model.config['dimension'],device=device)
        self.index=torch.zeros(1,dtype=torch.int64,device=device)
        self.temperature=torch.full((batch,),temperature,device=device)
        table=[]
        for j in range(steps):
            t=1-j/steps;tn=1-(j+1)/steps
            table.append([t,tn,(math.pi/2)*math.sin(math.pi*t/2),
                          (math.pi/2)*math.sin(math.pi*tn/2)])
        self.table=torch.tensor(table,dtype=torch.float32,device=device)
        self.steps=steps
        dt=-1/steps

        def step():
            values=self.table.index_select(0,self.index)[0]
            times=values[0].expand(batch);next_times=values[1].expand(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
                residual=model(self.x,times,self.temperature)
            drift=values[2]*residual.float()
            predictor=self.x+dt*drift
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
                residual2=model(predictor,next_times,self.temperature)
            self.x.copy_(self.x+.5*dt*(drift+values[3]*residual2.float()))
            self.index.add_(1)

        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(min(3,steps)):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        self.x.zero_();self.index.zero_()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            step()

    def sample(self,noise):
        self.x.copy_(noise);self.index.zero_()
        for _ in range(self.steps):
            self.graph.replay()
        return self.x.cpu()


@torch.no_grad()
def sample_latent(model,temperature,count,steps=400,batch=512,seed=7828,amp=False,cuda_graph=False):
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    results=[]
    model.eval()
    if hasattr(model,'configure_temperature'):
        model.configure_temperature(temperature)
    if cuda_graph and getattr(model,'_graph_temperature',None)!=temperature:
        model._sampling_graphs={}
        model._graph_temperature=temperature
    # For a=cos(pi*t/2), sigma=sin(pi*t/2), the probability-flow
    # ODE for score=-z-a*R is dz/dt=(pi/2)*sigma*R.
    # The analytic Gaussian drift cancels, avoiding a stiff linear ODE.
    for begin in range(0,count,batch):
        n=min(batch,count-begin)
        x=torch.randn(n,model.config['dimension'],device=device,generator=generator)
        if cuda_graph:
            key=(steps,n,amp)
            if key not in model._sampling_graphs:
                model._sampling_graphs[key]=CapturedHeun(model,temperature,steps,n,amp)
            results.append(model._sampling_graphs[key].sample(x))
            continue
        temps=torch.full((n,),temperature,device=device)
        for j in range(steps):
            t=1-j/steps;tn=1-(j+1)/steps;dt=-1/steps
            times=torch.full((n,),t,device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                r=model(x,times,temps)
            drift=(math.pi/2)*math.sin(math.pi*t/2)*r.float()
            predictor=x+dt*drift
            times.fill_(tn)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                r2=model(predictor,times,temps)
            drift2=(math.pi/2)*math.sin(math.pi*tn/2)*r2.float()
            x=x+.5*dt*(drift+drift2)
        results.append(x.cpu())
    return torch.cat(results).numpy()
