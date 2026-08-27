# cudaPIC - GPU-parallel PIC/MCC simulation (Rust-CUDA)

**cudaPIC** is an implementation of an RF plasma simulation using the
Particle-In-Cell method with Monte Carlo Collisions (PIC/MCC), in which the
entire computational cycle - particle motion, solving the Poisson equation,
charge density deposition, and collisions - is executed on the GPU. The
physical model is based on the reference, single-threaded eduPIC code
(Donko et al., 2021); however, this project is not a 1:1 port, but a
redesign of the algorithm for the SIMT architecture, written in Rust using
`cuda-oxide` and compiled directly
to PTX.

The description below focuses exclusively on the parallelization layer -
on *how* the sequential PIC/MCC algorithm was translated into CUDA
kernels, not on the physics of the simulation itself (which is described
in the original eduPIC publication).

## Data architecture: SoA + preallocation

Particles are stored in a *Structure of Arrays* layout
(`e_x/e_vx/e_vy/e_vz`, `i_x/i_vx/i_vy/i_vz`), rather than an array of
structures - this makes memory access fully coalesced when processing a
million threads. Buffers for electrons and ions are allocated upfront to
`MAX_PARTICLES` (a fixed GPU capacity), and the actual number of live
particles (`n_electrons`, `n_ions`) is a separate counter in device
memory - this avoids reallocation during the simulation, at the cost of
memory reserved in advance.

## CUDA kernels and parallelization strategies

### 1. Charge density deposition (`get_density`)
A classic PIC problem: many particles can land in the same grid node at
the same time. The solution is two-stage:
- each thread block accumulates partial contributions in **shared
  memory** (`SharedArray`) using a block-level atomic `fetch_add`
  (`BlockAtomicF32`),
- only after block synchronization is the result "flushed" once to the
  global grid via an atomic add (`DeviceAtomicF32`).

This reduces the number of costly atomic operations on global memory from
the order of the number of particles to the order of the number of
blocks × grid size.

### 2. Solving the Poisson equation (`solve_poisson_scan_f32`)
Instead of the classic Thomas elimination (a sequential algorithm that is
hard to parallelize), the potential on the grid is computed using the
**prefix-scan method (block_scan)**: the entire
tridiagonal system of size `N_G` is solved within a single block, as two
successive prefix sums (`block_scan::<f32, Sum, _>`), and the electric
field is computed from the resulting potential using a central
difference. The entire solver fits within a single kernel launch, with no
inter-block communication. An analogous, more general kernel,
`solve_poisson_dps_flexible` (with arbitrary boundary conditions), is used
to verify the correctness of the method (a convergence test against a
known analytical solution).

### 3. Particle motion + stream compaction (`move_and_compact`)
This is the most "GPU-native" part of the code. In a single pass:
- each thread advances its particle (leapfrog) and checks whether it has
  left the simulation domain,
- surviving particles are **compacted** into a dense output array without
  sorting - using warp-level voting (`warp::ballot`), prefix bit-counting
  within the mask (`lanemask_lt().count_ones()`), and a single-warp
  prefix scan over the partial warp sums, finished off with a single
  atomic `fetch_add` on the global live-particle counter.

The result: removing particles that struck an electrode happens without
transferring data to the CPU and without O(n log n) sorting - the entire
operation is linear and fully parallel.

### 4. Monte Carlo collisions (`check_collisions_e`, `check_collisions_i`)
Each thread independently decides whether its particle undergoes a
collision, and if so, what type of collision it is (elastic / excitation
/ ionization for electrons, isotropic / charge exchange for ions). The
key parallelism challenge: **ionization creates new particles** (a new
electron and a new ion) while the kernel is running. This is handled with
an atomic `fetch_add` on a dedicated "live particle" counter - every
thread that spawns a new particle gets a unique, collision-free write
index, with no need for any global barrier or a second pass.

### 5. Random number generator: per-thread xoshiro128+
Instead of a global RNG (a bottleneck and synchronization point), each
particle has its **own, independent stream** of the `xoshiro128+`
generator, initialized on the CPU using a jump function (`jump()`, an
algebraic skip of the generator by 2⁶⁴ steps) so that the streams of
millions of threads do not overlap. The entire generator - including the
Box-Muller transform for normal samples (`rng_next_three_normal`) - is
implemented directly in the kernel code, with no dependency on host-side
RNG libraries.

### 6. Math primitives in PTX
Transcendental functions (`sin`, `cos`, `sqrt`, `exp`, `atan2`, `acos`,
etc.) are invoked as hardware instructions in PTX
via inline
assembly (`ptx_asm!`). This design choice is the result of lack of support
for 30xx NVIDIA GPUs direct NVVM function calls.

### 7. Ion subcycling
Ions move `N_SUB` times more slowly than electrons on the timescale of
ion motion, so their kernels (`get_density`, `move_and_compact`,
`check_collisions_i`) are launched only every `N_SUB` electron time steps
(`t % N_SUB == 0`).

## CPU <-> GPU data flow

The programming model is designed so that **the entire simulation loop
runs on the GPU without per-step transfers**:
- particle data, cross sections, and RNG state are uploaded once, before
  the loop,
- the CPU only reads scalar particle counters from the GPU every
  `CHECKPOINT_CYCLES` (via `PinnedHostBuffer`, asynchronous transfer) to
  report progress,
- full results (particle positions/velocities, averaged densities, EEPF)
  are downloaded only once, at the end of the simulation.

## Requirements

- An NVIDIA GPU supporting CUDA,
- Rust toolchain + `cuda-oxide`to compile
  kernels to PTX.

## Running

```
cudaPIC <number_of_RF_cycles> [--measure] [number_of_measurement_cycles]
```

`--measure` enables accumulation of the averaged charge density and the
EEPF over a window of the last `number_of_measurement_cycles` RF cycles.

## Origin and license

The physical model and numerical scheme come from eduPIC 1.0
(Z. Donko, A. Derzsi, M. Vass, B. Horvath, S. Wilczek, B. Hartmann, and P. Hartmann, (c) 2021),
described in: Z. Donko et al., *Plasma Sources Science and Technology* 2021. 

This project (cudaPIC) is an independent reimplementation focused on GPU
parallelization and is not an official part of the eduPIC package, nor is
it supported by its authors. License: GNU GPL v3 - see the LICENSE file
or https://www.gnu.org/licenses/gpl-3.0.html.