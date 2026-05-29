//-------------------------------------------------------------------//
//       eduPIC-GPU : GPU-parallel 1d3v PIC/MCC simulation           //
//       Based on eduPIC by Z. Donko et al. (2021)                   //
//       Parallelized with cuda-oxide for NVIDIA GPUs                //
//-------------------------------------------------------------------//

#![allow(non_snake_case)]
#![allow(dead_code)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use std::env;
use std::time::Instant;

/// Simulation precision: change to f32 for single-precision GPU computation.
type Real = f64;

// constants

const PI: f64              = 3.141592653589793;      // mathematical constant Pi
const TWO_PI: f64          = 2.0 * PI;               // two times Pi
const E_CHARGE: f64        = 1.60217662e-19;         // electron charge [C]
const EV_TO_J: f64         = E_CHARGE;               // eV <-> Joule conversion factor
const E_MASS: f64          = 9.10938356e-31;         // mass of electron [kg]
const AR_MASS: f64         = 6.63352090e-26;         // mass of argon atom [kg]
const MU_ARAR: f64         = AR_MASS / 2.0;          // reduced mass of two argon atoms [kg]
const K_BOLTZMANN: f64     = 1.38064852e-23;         // Boltzmann's constant [J/K]
const EPSILON0: f64        = 8.85418781e-12;         // permittivity of free space [F/m]

// simulation parameters

const N_G: usize           = 400;                    // number of grid points
const N_T: u32             = 4000;                   // time steps within an RF period
const FREQUENCY: f64       = 13.56e6;                // driving frequency [Hz]
const VOLTAGE: f64         = 250.0;                  // voltage amplitude [V]
const L: f64               = 0.025;                  // electrode L [m]
const PRESSURE: f64        = 10.0;                   // gas pressure [Pa]
const TEMPERATURE: f64     = 350.0;                  // background gas temperature [K]
const WEIGHT: f64          = 7.0e4;                  // weight of superparticles
const ELECTRODE_AREA: f64  = 1.0e-4;                 // (fictive) electrode area [m^2]
const N_INIT: usize        = 1000;                   // number of initial electrons and ions

// additional (derived) constants

const PERIOD: f64          = 1.0 / FREQUENCY;                          // RF period length [s]
const DT_E: f64            = PERIOD / (N_T as f64);                    // electron time step [s]
const N_SUB: u32           = 20;                                       // ions move only in these cycles (subcycling)
const DT_I: f64            = (N_SUB as f64) * DT_E;                    // ion time step [s]
const DX: f64              = L / ((N_G - 1) as f64);                   // spatial grid division [m]
const INV_DX: f64          = 1.0 / DX;                                 // inverse of spatial grid size [1/m]
const GAS_DENSITY: f64     = PRESSURE / (K_BOLTZMANN * TEMPERATURE);   // background gas gas density [m-3]
const OMEGA: f64           = TWO_PI * FREQUENCY;                       // angular frequency [rad/s]

// electron and ion cross sections

const N_CS: usize          = 5;                      // total number of processes / cross sections
const E_ELA: usize         = 0;                      // process identifier: electron/elastic
const E_EXC: usize         = 1;                      // process identifier: electron/excitation
const E_ION: usize         = 2;                      // process identifier: electron/ionization
const I_ISO: usize         = 3;                      // process identifier: ion/elastic/isotropic
const I_BACK: usize        = 4;                      // process identifier: ion/elastic/backscattering
const E_EXC_TH: f64        = 11.5;                   // electron impact excitation threshold [eV]
const E_ION_TH: f64        = 15.8;                   // electron impact ionization threshold [eV]
const CS_RANGES: usize     = 1_000_000;              // number of entries in cross section arrays
const DE_CS: f64           = 0.001;                  // energy division in cross section arrays [eV]

// measurement conditions

const MIN_X: f64           = 0.45 * L;               // lower limit of central region
const MAX_X: f64           = 0.55 * L;               // upper limit of central region
const N_EEPF: usize        = 2000;                   // number of energy bins in Electron Energy Probability Function (EEPF)
const DE_EEPF: f64         = 0.05;                   // resolution of EEPF [eV]
const N_FED: usize         = 200;                    // number of energy bins in Flux-Energy Distributions (EFED and IFED)
const DE_FED: f64          = 1.0;                    // resolution of FEDs (EFED and IFED) [eV]
const N_BIN: u32           = 20;                     // number of time steps binned for the XT distributions
const N_XT: usize          = (N_T / N_BIN) as usize; // number of spatial bins for the XT distributions

// GPU capacity & allocation constants

const MAX_PARTICLES: usize = 120_000;                // maximum number of particles per species (pre-allocated on GPU).
const MAX_PARTICLES_U32: u32 = MAX_PARTICLES as u32; // used for LaunchConfig::for_num_elems
const N_SPECIES: usize        = 2;                    // electrons + ions
const PARTICLE_COMPS: usize   = 4;                    // arrays per species: x, vx, vy, vz
const N_GRID_ARRAYS: usize    = 5;                    // efield, pot, rho, e_density, i_density
const RNG_STATE_COMPS: usize  = 4;                    // xoshiro256** state: 4 × u64 per particle
const SIZEOF_REAL: usize      = std::mem::size_of::<Real>();  // adapts to Real precision
const BYTES_PER_MB: f64       = 1_048_576.0;          // 1024 × 1024

// cross section precomputation strategy:
// TODO: verify true branch
// - true:  sigma_tot = Σσ × v(E) × n_gas   - kernel just reads nu directly (no sqrt)
// - false: sigma_tot = Σσ × n_gas          - kernel must compute v and multiply (like original eduPIC)
const PRECOMPUTE_COLLISION_FREQ: bool = false;

const FACTOR_E: f64 = DT_E / E_MASS * (-E_CHARGE);  // leapfrog acceleration factor for electrons [m/s per (V/m)]
const FACTOR_I: f64 = DT_I / AR_MASS * E_CHARGE;    // leapfrog acceleration factor for ions [m/s per (V/m)]

// SoA particle data - host-side representation

struct ParticlesSoA {                                // Host-side SoA container for particle data.
    x:  Vec<Real>,
    vx: Vec<Real>,
    vy: Vec<Real>,
    vz: Vec<Real>,
}

impl ParticlesSoA {
    fn with_capacity(cap: usize) -> Self {
        Self {
            x:  vec![0.0 as Real; cap],
            vx: vec![0.0 as Real; cap],
            vy: vec![0.0 as Real; cap],
            vz: vec![0.0 as Real; cap],
        }
    }

    /// Return how many particles are currently valid (based on actual data length).
    fn len(&self) -> usize {
        self.x.len()
    }
}

// GPU buffer collection - all device-resident data

struct GpuSimState {
    // electron particle arrays (pre-allocated to MAX_PARTICLES)
    e_x:  DeviceBuffer<Real>,
    e_vx: DeviceBuffer<Real>,
    e_vy: DeviceBuffer<Real>,
    e_vz: DeviceBuffer<Real>,

    // ion particle arrays (pre-allocated to MAX_PARTICLES)
    i_x:  DeviceBuffer<Real>,
    i_vx: DeviceBuffer<Real>,
    i_vy: DeviceBuffer<Real>,
    i_vz: DeviceBuffer<Real>,

    // active particle counters (atomic on GPU)
    // pattern: pass as &[u32] to kernel, cast to DeviceAtomicU32 inside.
    // ionization appends directly to main arrays via atomicAdd on these counters.
    n_electrons: DeviceBuffer<u32>,  // n_electrons[0] = active electron count
    n_ions:      DeviceBuffer<u32>,  // n_ions[0] = active ion count

    // grid quantities (fixed size N_G = 400)
    efield:    DeviceBuffer<Real>,    // electric field [V/m]
    pot:       DeviceBuffer<Real>,    // electric potential [V]
    rho:       DeviceBuffer<Real>,    // charge density [C/m³]
    e_density: DeviceBuffer<Real>,    // electron density [m⁻³]
    i_density: DeviceBuffer<Real>,    // ion density [m⁻³]

    // cross sections (read-only after upload, 5 × CS_RANGES entries)
    // flattened 2D: cs[process][energy_index] → cs[process * CS_RANGES + energy_index]
    cs: DeviceBuffer<Real>,

    // total cross sections for null-collision method.
    // If PRECOMPUTE_COLLISION_FREQ: stores ν(E) = Σσ(E) × v(E) × n_gas  (kernel: nu = table[idx])
    // If !PRECOMPUTE_COLLISION_FREQ: stores Σσ(E) × n_gas               (kernel: nu = table[idx] * v)
    sigma_tot_e: DeviceBuffer<Real>,  // [CS_RANGES]
    sigma_tot_i: DeviceBuffer<Real>,  // [CS_RANGES]

    // RNG state per-particle
    // flattened: rng_state[particle_idx * 4 + component]
    rng_state: DeviceBuffer<u64>,
}

impl GpuSimState {
    // allocate all GPU buffers (zeroed)
    fn allocate(stream: &cuda_core::CudaStream) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            // particle arrays
            e_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,

            i_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,

            // atomic counters
            n_electrons: DeviceBuffer::<u32>::zeroed(stream, 1)?,
            n_ions:      DeviceBuffer::<u32>::zeroed(stream, 1)?,

            // grid arrays
            efield:    DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            pot:       DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            rho:       DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            e_density: DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            i_density: DeviceBuffer::<Real>::zeroed(stream, N_G)?,

            // cross sections: flattened [N_CS × CS_RANGES]
            cs: DeviceBuffer::<Real>::zeroed(stream, N_CS * CS_RANGES)?,

            // total cross sections
            sigma_tot_e: DeviceBuffer::<Real>::zeroed(stream, CS_RANGES)?,
            sigma_tot_i: DeviceBuffer::<Real>::zeroed(stream, CS_RANGES)?,

            // RNG state: 4 × u64 per particle
            // TODO xoshiro256** - verify
            rng_state: DeviceBuffer::<u64>::zeroed(stream, MAX_PARTICLES * 4)?,
        })
    }

    // upload initial particle data from host.
    // only positions up to `n_active` are valid; rest stays zeroed.
    fn upload_electrons(
        &mut self,
        stream: &cuda_core::CudaStream,
        particles: &ParticlesSoA,
        n_active: u32,
    ) -> Result<(), cuda_core::DriverError> {
        self.e_x  = DeviceBuffer::from_host(stream, &particles.x)?;
        self.e_vx = DeviceBuffer::from_host(stream, &particles.vx)?;
        self.e_vy = DeviceBuffer::from_host(stream, &particles.vy)?;
        self.e_vz = DeviceBuffer::from_host(stream, &particles.vz)?;

        self.n_electrons = DeviceBuffer::from_host(stream, &[n_active])?;
        Ok(())
    }

    fn upload_ions(
        &mut self,
        stream: &cuda_core::CudaStream,
        particles: &ParticlesSoA,
        n_active: u32,
    ) -> Result<(), cuda_core::DriverError> {
        self.i_x  = DeviceBuffer::from_host(stream, &particles.x)?;
        self.i_vx = DeviceBuffer::from_host(stream, &particles.vx)?;
        self.i_vy = DeviceBuffer::from_host(stream, &particles.vy)?;
        self.i_vz = DeviceBuffer::from_host(stream, &particles.vz)?;

        self.n_ions = DeviceBuffer::from_host(stream, &[n_active])?;
        Ok(())
    }

    // upload cross section tables (flattened 2D) and total cross sections
    fn upload_cross_sections(
        &mut self,
        stream: &cuda_core::CudaStream,
        cs_flat: &[Real],        // [N_CS * CS_RANGES]
        sigma_tot_e: &[Real],    // [CS_RANGES]
        sigma_tot_i: &[Real],    // [CS_RANGES]
    ) -> Result<(), cuda_core::DriverError> {
        self.cs          = DeviceBuffer::from_host(stream, cs_flat)?;
        self.sigma_tot_e = DeviceBuffer::from_host(stream, sigma_tot_e)?;
        self.sigma_tot_i = DeviceBuffer::from_host(stream, sigma_tot_i)?;
        Ok(())
    }

    // upload RNG seeds
    fn upload_rng_state(
        &mut self,
        stream: &cuda_core::CudaStream,
        seeds: &[u64],
    ) -> Result<(), cuda_core::DriverError> {
        self.rng_state = DeviceBuffer::from_host(stream, seeds)?;
        Ok(())
    }

    // download electron data back to host
    fn download_electrons(
        &self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(ParticlesSoA, u32), cuda_core::DriverError> {
        let x  = self.e_x.to_host_vec(stream)?;
        let vx = self.e_vx.to_host_vec(stream)?;
        let vy = self.e_vy.to_host_vec(stream)?;
        let vz = self.e_vz.to_host_vec(stream)?;
        let n  = self.n_electrons.to_host_vec(stream)?;

        Ok((ParticlesSoA { x, vx, vy, vz }, n[0]))
    }

    // download ion data back to host
    fn download_ions(
        &self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(ParticlesSoA, u32), cuda_core::DriverError> {
        let x  = self.i_x.to_host_vec(stream)?;
        let vx = self.i_vx.to_host_vec(stream)?;
        let vy = self.i_vy.to_host_vec(stream)?;
        let vz = self.i_vz.to_host_vec(stream)?;
        let n  = self.n_ions.to_host_vec(stream)?;

        Ok((ParticlesSoA { x, vx, vy, vz }, n[0]))
    }
}

// GPU Kernels

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn move_particles(
        efield:   &[Real],
        mut x:    DisjointSlice<Real>,
        mut vx:   DisjointSlice<Real>,
        n_active: u32,
        factor:   Real,
        dt:       Real,
    ) {
        // E-field loaded cooperatively into shared memory once per block,
        // eliminating repeated global memory reads.
        static mut EFIELD_SHARED: SharedArray<Real, N_G> = SharedArray::UNINIT;

        let tid        = thread::threadIdx_x() as usize;
        let block_size = thread::blockDim_x()  as usize;

        // Each thread loads ceil(N_G / block_size) elements.
        let mut k = tid;
        while k < N_G {
            unsafe { EFIELD_SHARED[k] = efield[k]; }
            k += block_size;
        }
        thread::sync_threads();

        if let Some((x_val, idx)) = x.get_mut_indexed() {
            let i = idx.get();
            if i >= n_active as usize {
                return;
            }

            if let Some(vx_val) = vx.get_mut(idx) {
                let pos = *x_val * INV_DX as Real;
                let p   = pos as usize;                
                let c2  = pos - p as Real;
                let e_x = unsafe {
                    (1.0 as Real - c2) * EFIELD_SHARED[p]
                        + c2 * EFIELD_SHARED[p + 1]
                };

                let new_vx = *vx_val + factor * e_x;
                *vx_val = new_vx;
                *x_val  = *x_val + new_vx * dt;
            }
        }
    }

    // TODO - no shmem perf testing
    #[kernel]
    pub fn OLD_move_particles(
        efield:   &[Real],
        mut x:    DisjointSlice<Real>,
        mut vx:   DisjointSlice<Real>,
        n_active: u32,
        factor:   Real,
        dt:       Real,
    ) {
        if let Some((x_val, idx)) = x.get_mut_indexed() {
            let i = idx.get();
            if i >= n_active as usize {
                return;
            }
            if let Some(vx_val) = vx.get_mut(idx) {
                let pos = *x_val * INV_DX as Real;
                let p   = pos as usize;                
                let c2  = pos - p as Real;
                let e_x = (1.0 as Real - c2) * efield[p]
                        + c2 * efield[p + 1];

                let new_vx = *vx_val + factor * e_x;
                *vx_val = new_vx;
                *x_val  = *x_val + new_vx * dt;
            }
        }
    }

    // TODO: deposit_charge (density accumulation with atomics / ...)
    // TODO: solve_poisson (parallel tridiagonal / prefix-sum solver / ...)
    // TODO: check_boundaries (stream compaction / ...)
    // TODO: collisions_e (electron-neutral MCC, ionization appends directly / ...)
    // TODO: collisions_i (ion-neutral MCC / ...)
}

// Host-side initialization helpers

fn init_cross_sections() -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    let mut cs_flat = vec![0.0 as Real; N_CS * CS_RANGES];
    let mut sigma_tot_e = vec![0.0 as Real; CS_RANGES];
    let mut sigma_tot_i = vec![0.0 as Real; CS_RANGES];

    // Cross-section formulas always computed in f64 for numerical accuracy,
    // then stored as Real for GPU consumption. // TODO - verify if this is necessary or if we can compute directly in Real (f32) for single-precision GPU.
    let qmom = |e: f64| -> f64 {
        1.0e-20*(
        (6.0/(1.0+e/0.1+(e/0.6).powf(2.0)).powf(3.3)-1.1*e.powf(1.4)/
        (1.0+(e/15.0).powf(1.2))/(1.0+(e/5.5).powf(2.5)+(e/60.0).powf(4.1)).sqrt()).abs()+0.05/(1.0+e/10.0).powf(2.0)+
        0.01*e.powf(3.0)/(1.0+(e/12.0).powf(6.0)))
    };

    let qexc = |e: f64| -> f64 {
        if e <= E_EXC_TH{0.0} else {(0.034 * (e - 11.5).powf(1.1) * (1.0 + (e / 15.0).powf(2.8))
        / (1.0 + (e / 23.0).powf(5.5)) + 0.023 * (e - 11.5) / (1.0 + e / 80.0).powf(1.9))*1.0e-20 }
    };

    let qion = |e: f64| -> f64 {
        if e <= E_ION_TH{0.0} else {(970.0 * (e - 15.8) / (70.0 + e).powf(2.0)
        + 0.06 * (e - 15.8).powf(2.0) * (-e / 9.0).exp())*1.0e-20 }
    };

    let qmoi = |e_lab: f64| -> f64 {
        1.15e-18 * e_lab.powf(-0.1) * (1.0 + 0.015 / e_lab).powf(0.6) //2*e!
    };
    let qiso = |e_lab: f64| -> f64 {
        2.0e-19 * e_lab.powf(-0.5) / (1.0 + e_lab) + 3.0e-19 * e_lab / (1.0 + e_lab / 3.0).powf(2.0)
    };
    let qchx = |e_lab: f64| -> f64 { 0.5*(qmoi(e_lab)-qiso(e_lab)) };

    for i in 0..CS_RANGES {
        let e = if i == 0 { DE_CS } else { (i as f64) * DE_CS };

        cs_flat[E_ELA * CS_RANGES + i] = qmom(e) as Real;
        cs_flat[E_EXC * CS_RANGES + i] = qexc(e) as Real;
        cs_flat[E_ION * CS_RANGES + i] = qion(e) as Real;
        cs_flat[I_ISO * CS_RANGES + i] = qiso(2.0 * e) as Real;
        cs_flat[I_BACK * CS_RANGES + i] = qchx(2.0 * e) as Real;
    }

    for i in 0..CS_RANGES {
        let e = if i == 0 { DE_CS } else { (i as f64) * DE_CS };

        let sum_e = cs_flat[E_ELA * CS_RANGES + i] as f64
                  + cs_flat[E_EXC * CS_RANGES + i] as f64
                  + cs_flat[E_ION * CS_RANGES + i] as f64;

        let sum_i = cs_flat[I_ISO * CS_RANGES + i] as f64
                  + cs_flat[I_BACK * CS_RANGES + i] as f64;

        // TODO: verify true branch
        if PRECOMPUTE_COLLISION_FREQ {
            let v_e = (2.0 * e * EV_TO_J / E_MASS).sqrt();
            let v_i = (2.0 * e * EV_TO_J / MU_ARAR).sqrt();
            sigma_tot_e[i] = (sum_e * v_e * GAS_DENSITY) as Real;
            sigma_tot_i[i] = (sum_i * v_i * GAS_DENSITY) as Real;
        } else {
            sigma_tot_e[i] = (sum_e * GAS_DENSITY) as Real;
            sigma_tot_i[i] = (sum_i * GAS_DENSITY) as Real;
        }
    }

    (cs_flat, sigma_tot_e, sigma_tot_i)
}

/// generate xoshiro256** seed state for all particle slots.
/// each particle gets 4 × u64 seeded from a master RNG.
fn generate_rng_seeds(n: usize) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut seeds = vec![0u64; n * 4];
    for i in 0..n {
        for j in 0..4 {
            let mut hasher = DefaultHasher::new();
            (i as u64 * 4 + j as u64).hash(&mut hasher);
            seeds[i * 4 + j] = hasher.finish() | 1;       // ensure non-zero state (xoshiro requirement)
        }
    }
    seeds
}

fn main() {
    perform_tests();

    println!(">> eduPIC-GPU: starting...");
    println!(">> eduPIC-GPU: cuda-oxide parallel PIC/MCC simulation");
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: edupic-gpu <num_cycles>");
        std::process::exit(1);
    }
    let num_cycles: usize = args[1].parse().expect("Invalid cycle count");

    let start = Instant::now();

    // 1. Initialize CUDA context
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context (no GPU?)");
    let stream = ctx.default_stream();
    println!(">> eduPIC-GPU: CUDA context initialized");

    // 2. Compute cross-sections on CPU
    let (cs_flat, sigma_tot_e, sigma_tot_i) = init_cross_sections();
    println!(">> eduPIC-GPU: cross-sections computed ({} entries per process)", CS_RANGES);

    // 3. Initialize particles on CPU (SoA layout)
    let mut electrons_host = ParticlesSoA::with_capacity(MAX_PARTICLES);
    let mut ions_host = ParticlesSoA::with_capacity(MAX_PARTICLES);

    // Populate first N_INIT particles (uniform random positions, thermal velocities)
    // TODO: replace with proper initialization from original eduPIC logic
    let n_init_active = N_INIT as u32;

    // 4. Allocate all GPU buffers
    let mut gpu = GpuSimState::allocate(&stream)
        .expect("Failed to allocate GPU memory");
    println!(">> eduPIC-GPU: GPU memory allocated (~{:.1} MB)",
        (MAX_PARTICLES * PARTICLE_COMPS * SIZEOF_REAL * N_SPECIES  // particles (e + i)
        + N_CS * CS_RANGES * SIZEOF_REAL                            // cross sections
        + CS_RANGES * SIZEOF_REAL * N_SPECIES                       // sigma_tot_e + sigma_tot_i
        + N_G * N_GRID_ARRAYS * SIZEOF_REAL                         // grid arrays
        + MAX_PARTICLES * RNG_STATE_COMPS * 8                       // RNG state (always u64)
        ) as f64 / BYTES_PER_MB
    );

    // 5. Upload data to GPU (one-time PCIe transfer)
    gpu.upload_electrons(&stream, &electrons_host, n_init_active)
        .expect("Failed to upload electrons");
    gpu.upload_ions(&stream, &ions_host, n_init_active)
        .expect("Failed to upload ions");
    gpu.upload_cross_sections(&stream, &cs_flat, &sigma_tot_e, &sigma_tot_i)
        .expect("Failed to upload cross-sections");

    let rng_seeds = generate_rng_seeds(MAX_PARTICLES);
    gpu.upload_rng_state(&stream, &rng_seeds)
        .expect("Failed to upload RNG state");

    println!(">> eduPIC-GPU: data uploaded to GPU");

    // 6. Launch config - fixed for entire simulation (zero-sync pattern)
    let cfg = LaunchConfig::for_num_elems(MAX_PARTICLES_U32);

    // 7. GPU simulation loop
    // all kernels launched on same stream
    println!(">> eduPIC-GPU: running {} cycles × {} steps (GPU-resident)...", num_cycles, N_T);
    let module = kernels::load(&ctx).expect("Failed to load CUDA module");
    
    for _cycle in 0..num_cycles {
        for _t in 0..N_T {
            module.move_particles(&stream, cfg,
                &gpu.efield, &mut gpu.e_x, &mut gpu.e_vx,
                n_init_active, FACTOR_E as Real, DT_E as Real,
            ).expect("move_particles (electrons) failed");
            // module.deposit_charge_e(&stream, cfg, ...)?;
            // module.deposit_charge_i(&stream, cfg, ...)?;
            // module.solve_poisson(&stream, poisson_cfg, ...)?;
            // if _t % N_SUB == 0 {
            //     module.move_particles(&stream, cfg,
            //         &gpu.efield, &mut gpu.i_x, &mut gpu.i_vx,
            //         n_init_active, FACTOR_I as Real, DT_I as Real,
            //     ).expect("move_particles (ions) failed");
            // }
            // module.check_boundaries_e(&stream, cfg, ...)?;
            // module.check_boundaries_i(&stream, cfg, ...)?;
            // module.collisions_e(&stream, cfg, ...)?;  // ionization appends directly
            // if _t % N_SUB == 0 { module.collisions_i(&stream, cfg, ...)?; }
        }
    }

    // 8. Synchronize and download results
    ctx.synchronize().expect("CUDA synchronization failed");

    let (electrons_result, n_e_final) = gpu.download_electrons(&stream)
        .expect("Failed to download electrons");
    let (ions_result, n_i_final) = gpu.download_ions(&stream)
        .expect("Failed to download ions");

    let elapsed = start.elapsed().as_secs_f64();
    println!(">> eduPIC-GPU: simulation complete in {:.3} s", elapsed);
    println!(">> eduPIC-GPU: final particles: {} electrons, {} ions", n_e_final, n_i_final);
}

// tests

fn perform_tests() {
    test_move_particles_analytic();
    test_move_particles_edge_cases();
    test_move_particles();
    bench_shmem_vs_no_shmem();
}

// expected result is computable without oracle
fn test_move_particles_analytic() {
    const E_UNIFORM: Real = 100.0;
    let efield_host = vec![E_UNIFORM; N_G];

    let n: usize = 10;
    let x_host:  Vec<Real> = (0..n).map(|i| L as Real * (i as Real + 0.5) / n as Real).collect();
    let vx_host: Vec<Real> = (0..n).map(|i| (i as Real - n as Real / 2.0) * 1e5).collect();

    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let efield_dev  = DeviceBuffer::from_host(&stream, &efield_host).unwrap();
    let mut x_dev   = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut vx_dev  = DeviceBuffer::from_host(&stream, &vx_host).unwrap();
    let module = kernels::load(&ctx).unwrap();
    let cfg    = LaunchConfig::for_num_elems(n as u32);
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        n as u32, FACTOR_E as Real, DT_E as Real,
    ).unwrap();

    let x_gpu  = x_dev.to_host_vec(&stream).unwrap();
    let vx_gpu = vx_dev.to_host_vec(&stream).unwrap();

    // analytic expected values:
    //   new_vx = vx_0 + FACTOR_E * E_UNIFORM
    //   new_x  = x_0  + new_vx * DT_E
    let eps: Real = if std::mem::size_of::<Real>() == 4 { 1e-5 as Real } else { 1e-10 as Real };
    let mut errors = 0;
    for i in 0..n {
        let expected_vx = vx_host[i] + FACTOR_E as Real * E_UNIFORM;
        let expected_x  = x_host[i]  + expected_vx * DT_E as Real;
        if (vx_gpu[i] - expected_vx).abs() > eps {
            eprintln!("analytic vx[{}]: got={:.15e} expected={:.15e}", i, vx_gpu[i], expected_vx);
            errors += 1;
        }
        if (x_gpu[i] - expected_x).abs() > eps {
            eprintln!("analytic x[{}]: got={:.15e} expected={:.15e}", i, x_gpu[i], expected_x);
            errors += 1;
        }
    }
    if errors > 0 {
        println!("test_move_particles_analytic: {} mismatches", errors);
        std::process::exit(1);
    }
}

fn test_move_particles_edge_cases() {
    let efield_host: Vec<Real> = (0..N_G).map(|i| (i as Real + 1.0) * 50.0).collect();

    // case A: particle exactly on a grid node (c2 = 0, e_x = efield[p] exactly)
    // case B: particle near right boundary
    // case C: zero velocity particle (new_x changes only from field acceleration)
    let x_cases:  Vec<Real> = vec![
        10.0 * DX as Real,                      // A: exactly on node 10
        398.5 * DX as Real,                     // B: near right boundary
        L as Real * 0.5,                        // C: midpoint, vx = 0
    ];
    let vx_cases: Vec<Real> = vec![
        1e4,   // A
        1e4,   // B
        0.0,   // C: zero velocity
    ];
    let n = x_cases.len();

    // CPU oracle
    let mut x_cpu  = x_cases.clone();
    let mut vx_cpu = vx_cases.clone();
    for i in 0..n {
        let p   = (x_cpu[i] * INV_DX as Real) as usize;
        let c2  = x_cpu[i] * INV_DX as Real - p as Real;
        let e_x = (1.0 as Real - c2) * efield_host[p] + c2 * efield_host[p + 1];
        vx_cpu[i] = vx_cpu[i] + FACTOR_E as Real * e_x;
        x_cpu[i]  = x_cpu[i]  + vx_cpu[i] * DT_E as Real;
    }

    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let efield_dev  = DeviceBuffer::from_host(&stream, &efield_host).unwrap();
    let mut x_dev   = DeviceBuffer::from_host(&stream, &x_cases.clone()).unwrap();
    let mut vx_dev  = DeviceBuffer::from_host(&stream, &vx_cases).unwrap();
    let module = kernels::load(&ctx).unwrap();
    let cfg    = LaunchConfig::for_num_elems(n as u32);
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        n as u32, FACTOR_E as Real, DT_E as Real,
    ).unwrap();

    let x_gpu  = x_dev.to_host_vec(&stream).unwrap();
    let vx_gpu = vx_dev.to_host_vec(&stream).unwrap();

    let eps: Real = if std::mem::size_of::<Real>() == 4 { 1e-5 as Real } else { 1e-10 as Real };
    let labels = ["grid_node", "near_boundary", "zero_vx"];
    let mut errors = 0;
    for i in 0..n {
        if (x_gpu[i] - x_cpu[i]).abs() > eps {
            eprintln!("edge[{}] x:  GPU={:.15e} CPU={:.15e}", labels[i], x_gpu[i], x_cpu[i]);
            errors += 1;
        }
        if (vx_gpu[i] - vx_cpu[i]).abs() > eps {
            eprintln!("edge[{}] vx: GPU={:.15e} CPU={:.15e}", labels[i], vx_gpu[i], vx_cpu[i]);
            errors += 1;
        }
    }
    if errors > 0 {
        println!("test_move_particles_edge_cases: {} mismatches", errors);
        std::process::exit(1);
    }
}

fn test_move_particles() {
    let n_test: usize = 1000;
    let mut x_host  = vec![0.0 as Real; n_test];
    let mut vx_host = vec![0.0 as Real; n_test];
    let efield_host: Vec<Real> = (0..N_G).map(|i| 100.0 * (i as Real / N_G as Real)).collect();

    for i in 0..n_test {
        x_host[i] = (L * (i as f64 + 0.5) / n_test as f64) as Real;
        vx_host[i] = (1000.0 * (i as f64 - n_test as f64 / 2.0)) as Real;
    }

    // CPU oracle
    let mut x_cpu = x_host.clone();
    let mut vx_cpu = vx_host.clone();
    for i in 0..n_test {
        let p = (x_cpu[i] * INV_DX as Real) as usize;
        let c2 = x_cpu[i] * INV_DX as Real - p as Real;
        let e_x = (1.0 as Real - c2) * efield_host[p] + c2 * efield_host[p + 1];
        vx_cpu[i] += FACTOR_E as Real * e_x;
        x_cpu[i] += vx_cpu[i] * DT_E as Real;
    }

    // GPU execution
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let efield_dev = DeviceBuffer::from_host(&stream, &efield_host).unwrap();
    let mut x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut vx_dev = DeviceBuffer::from_host(&stream, &vx_host).unwrap();
    let module = kernels::load(&ctx).unwrap();
    let cfg = LaunchConfig::for_num_elems(n_test as u32);
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        n_test as u32,
        FACTOR_E as Real,
        DT_E as Real,
    ).unwrap();

    // compare
    let x_gpu = x_dev.to_host_vec(&stream).unwrap();
    let vx_gpu = vx_dev.to_host_vec(&stream).unwrap();

    let eps: Real = if std::mem::size_of::<Real>() == 4 { 1e-5 as Real } else { 1e-10 as Real };
    let mut errors = 0;
    for i in 0..n_test {
        if (x_gpu[i] - x_cpu[i]).abs() > eps {
            eprintln!("x[{}]: GPU={:.15e} CPU={:.15e} diff={:.2e}", i, x_gpu[i], x_cpu[i], (x_gpu[i]-x_cpu[i]).abs());
            errors += 1;
        }
        if (vx_gpu[i] - vx_cpu[i]).abs() > eps {
            eprintln!("vx[{}]: GPU={:.15e} CPU={:.15e} diff={:.2e}", i, vx_gpu[i], vx_cpu[i], (vx_gpu[i]-vx_cpu[i]).abs());
            errors += 1;
        }
    }

    if errors > 0 {
        println!("move_particles: {} mismatches", errors);
        std::process::exit(1);
    }
}

// timing benchmark: shared-memory vs. no-shared-memory particle pusher
fn bench_shmem_vs_no_shmem() {
    const N: usize     = MAX_PARTICLES;
    const REPS: u32    = 1000;

    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();
    let cfg    = LaunchConfig::for_num_elems(N as u32);

    // random positions across [0, L], random velocities
    let efield_host: Vec<Real> = (0..N_G).map(|i| 100.0 * (i as Real / N_G as Real)).collect();
    let x_init:  Vec<Real> = (0..N).map(|i| L as Real * (i as Real / N as Real)).collect();
    let vx_init: Vec<Real> = (0..N).map(|i| 1000.0 * ((i as Real) - N as Real / 2.0)).collect();

    let efield_dev       = DeviceBuffer::from_host(&stream, &efield_host).unwrap();
    let mut x_shmem      = DeviceBuffer::from_host(&stream, &x_init).unwrap();
    let mut vx_shmem     = DeviceBuffer::from_host(&stream, &vx_init).unwrap();
    let mut x_no_shmem   = DeviceBuffer::from_host(&stream, &x_init).unwrap();
    let mut vx_no_shmem  = DeviceBuffer::from_host(&stream, &vx_init).unwrap();

    let factor = FACTOR_E as Real;
    
    // dt = 0 freezes positions across launches so particles never leave [0, L].
    let dt = 0.0 as Real;

    // warm-up (avoids JIT / driver overhead in measurements) TODO - verify if needed
    for _ in 0..10 {
        module.move_particles(&stream, cfg,
            &efield_dev, &mut x_shmem, &mut vx_shmem, N as u32, factor, dt).unwrap();
        module.OLD_move_particles(&stream, cfg,
            &efield_dev, &mut x_no_shmem, &mut vx_no_shmem, N as u32, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();

    // time shared-memory variant
    let t0 = Instant::now();
    for _ in 0..REPS {
        module.move_particles(&stream, cfg,
            &efield_dev, &mut x_shmem, &mut vx_shmem, N as u32, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();
    let t_shmem = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6; // µs per launch

    // time no-shared-memory variant
    let t0 = Instant::now();
    for _ in 0..REPS {
        module.OLD_move_particles(&stream, cfg,
            &efield_dev, &mut x_no_shmem, &mut vx_no_shmem, N as u32, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();
    let t_no_shmem = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6;

    println!(">> bench move_particles ({} particles, {} reps):", N, REPS);
    println!("     shared memory : {:.2} µs/launch", t_shmem);
    println!("     no shared mem : {:.2} µs/launch", t_no_shmem);
    println!("     speedup       : {:.2}×", t_no_shmem / t_shmem);
}
