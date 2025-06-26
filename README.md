# FastTD3X: `jax` port of FastTD3

> "AMP and torch.compile While JAX-based RL implementations have become popular in recent days for its speed, we build our implementation upon PyTorch (Paszke et al., 2019) for its simplicity and flexibility."

- They don't  fuse the entire simulation-to-update pipeline into a single accelearted graph
    - There is a bunch of missing execution-level fusion benefits that are missing

#### Performance enhancement limited to AMP + torch.compile

The fastest configuration combines automatic mixed precision (AMP) and PyTorch’s torch.compile, yielding up to 70% speedup, but no further:

> “mixed-precision training with AMP and bfloat16 accelerates training by up to 40% … torch.compile … up to a 35% speedup … combined … up to 70%.” 
This remains far below full JIT/XLA pipelines or GPU-native simulation reimplementations.

## Other Pure Jax Implementations claim to be orders-of-magnitude faster

### PureJaxRl's Benchmarks

Without vectorization, our implementation runs 10x faster than [CleanRL's PyTorch baselines](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), as shown in the single-thread performance plot.

Cartpole                   |  Minatar-Breakout
:-------------------------:|:-------------------------:
![](https://github.com/luchris429/purejaxrl/raw/main/docs/cartpole_plot_seconds.png)  |  ![](https://github.com/luchris429/purejaxrl/raw/main/docs/minatar_plot_seconds.png)


With vectorized training, we can train 2048 PPO agents in half the time it takes to train a single PyTorch PPO agent on a single GPU. The vectorized agent training allows for simultaneous training across multiple seeds, rapid hyperparameter tuning, and even evolutionary Meta-RL. 

Vectorised Cartpole        |  Vectorised Minatar-Breakout
:-------------------------:|:-------------------------:
![](https://github.com/luchris429/purejaxrl/raw/main/docs/cartpole_plot_parallel.png)  |  ![](https://github.com/luchris429/purejaxrl/raw/main/docs/minatar_plot_parallel.png)

For more, see [this blog post](https://chrislu.page/blog/meta-disco/). 

Also, there is explicit data type conversions and CPU-GPU transfers/memory allocation inefficiencies that could be avoided with a pure Jax implementaiton.

## Rough Plan
- Reimplement with Jax/Flax
- Integrate with Mujoco playground
- Benchmark results against normal FastTD3
- Possibly integrate into our framework? Otherwise would be useful to others as an open-source thing. Maybe could be integrated into other frameworks like PureJaxRL
- Maybe write a small blog post about this explaining FastTD3 and the optimizations

