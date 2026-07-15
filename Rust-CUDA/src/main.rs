//-------------------------------------------------------------------//
//       eduPIC-GPU : GPU-parallel 1d3v PIC/MCC simulation           //
//       Based on eduPIC by Z. Donko et al. (2021)                   //
//       Parallelized with cuda-oxide for NVIDIA GPUs                //
//-------------------------------------------------------------------//

#![allow(non_snake_case)]
#![allow(dead_code)]

use cuda_core::{memory, CudaContext, DeviceBuffer, PinnedHostBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, ptx_asm, DisjointSlice, SharedArray};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicF32, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::cooperative_groups::{block_scan, ops::Sum, this_thread_block};
use rand::RngExt;
use rand_distr::Normal;
use std::{env, fmt};
use std::time::Instant;
use std::io::BufWriter;


/// TODO - Simulation precision: change to f32 for single-precision GPU computation.
type Real = f32;

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
const SIZEOF_REAL: usize      = std::mem::size_of::<Real>();  // adapts to Real precision
const BYTES_PER_MB: f64       = 1_048_576.0;          // 1024 × 1024

// cross section precomputation strategy:
// TODO: verify true branch
// - true:  sigma_tot = Σσ × v(E) × n_gas   - kernel just reads nu directly (no sqrt)
// - false: sigma_tot = Σσ × n_gas          - kernel must compute v and multiply (like original eduPIC)
const PRECOMPUTE_COLLISION_FREQ: bool = false;

const FACTOR_E: f64 = DT_E / E_MASS * (-E_CHARGE);  // leapfrog acceleration factor for electrons [m/s per (V/m)]
const FACTOR_I: f64 = DT_I / AR_MASS * E_CHARGE;    // leapfrog acceleration factor for ions [m/s per (V/m)]
const WEIGHT_FACTOR: f64 = WEIGHT / (ELECTRODE_AREA * DX);

// block must have ≥ N_G threads so each thread owns one grid point
// Next multiple of 32 above N_G=400 is 416, but 512 is cleaner (2 warps of power-of-two).
const SCAN_BLOCK_SIZE: u32   = 512;                             // 16 warps, covers N_G=400
const SCAN_NUM_WARPS:  usize = SCAN_BLOCK_SIZE as usize / 32;   

// check_boundaries stream compaction: per-block prefix scan + 1 atomicAdd/block.
// Block size is fixed at 256 (8 warps) to match LaunchConfig::for_num_elems. TODO - verify
const COMPACT_BLOCK_SIZE: u32   = 256;
const COMPACT_NUM_WARPS:  usize = COMPACT_BLOCK_SIZE as usize / 32;   // = 8

// collisions
const NORMAL_RANGE: Real = 269.90040554976775; // (K_BOLTZMANN * TEMPERATURE / AR_MASS).sqrt();  // thermal velocity of background gas [m/s]
const F1: Real = (E_MASS / (E_MASS + AR_MASS)) as Real;
const F2: Real = (AR_MASS / (E_MASS + AR_MASS)) as Real;
const LOG2_E: Real = 1.4426950408889634; // log2(e)

// precomputed
const HALF_E_MASS_OVER_E_CHARGE: Real = (0.5 * E_MASS / E_CHARGE) as Real;
const HALF_MU_ARAR_OVER_E_CHARGE: Real = (0.5 * MU_ARAR / E_CHARGE) as Real;

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
    rng_e0: DeviceBuffer<u32>,
    rng_e1: DeviceBuffer<u32>,
    rng_e2: DeviceBuffer<u32>,
    rng_e3: DeviceBuffer<u32>,
    rng_i0: DeviceBuffer<u32>,
    rng_i1: DeviceBuffer<u32>,
    rng_i2: DeviceBuffer<u32>,
    rng_i3: DeviceBuffer<u32>,

    // double-buffer pattern for check_boundaries stream compaction
    tmp_x:  DeviceBuffer<Real>,
    tmp_vx: DeviceBuffer<Real>,
    tmp_vy: DeviceBuffer<Real>,
    tmp_vz: DeviceBuffer<Real>,

    // alive counter for stream compaction
    alive_counter: DeviceBuffer<u32>,
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

            // RNG state: 4 × u32 per particle
            // TODO xoshiro128+ - verify
            rng_e0: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e1: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e2: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e3: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i0: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i1: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i2: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i3: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,

            // tmp buffers for stream compaction
            tmp_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            alive_counter: DeviceBuffer::<u32>::zeroed(stream, 1)?,
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
        self.e_x.copy_from_host(stream, &particles.x)?;
        self.e_vx.copy_from_host(stream, &particles.vx)?;
        self.e_vy.copy_from_host(stream, &particles.vy)?;
        self.e_vz.copy_from_host(stream, &particles.vz)?;

        self.n_electrons.copy_from_host(stream, &[n_active])?;
        Ok(())
    }

    fn upload_ions(
        &mut self,
        stream: &cuda_core::CudaStream,
        particles: &ParticlesSoA,
        n_active: u32,
    ) -> Result<(), cuda_core::DriverError> {
        self.i_x.copy_from_host(stream, &particles.x)?;
        self.i_vx.copy_from_host(stream, &particles.vx)?;
        self.i_vy.copy_from_host(stream, &particles.vy)?;
        self.i_vz.copy_from_host(stream, &particles.vz)?;

        self.n_ions.copy_from_host(stream, &[n_active])?;
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
        self.cs.copy_from_host(stream, cs_flat)?;
        self.sigma_tot_e.copy_from_host(stream, sigma_tot_e)?;
        self.sigma_tot_i.copy_from_host(stream, sigma_tot_i)?;
        Ok(())
    }

    // upload RNG seeds
    fn upload_rng_state(
        &mut self,
        stream: &cuda_core::CudaStream,
        e_seeds: &[[u32; 4]],
        i_seeds: &[[u32; 4]],
    ) -> Result<(), cuda_core::DriverError> {
        fn split(src: &[[u32; 4]]) -> [Vec<u32>; 4] {
            core::array::from_fn(|k| src.iter().map(|s| s[k]).collect())
        }
        let [s0, s1, s2, s3] = split(e_seeds);
        self.rng_e0.copy_from_host(stream, &s0)?;
        self.rng_e1.copy_from_host(stream, &s1)?;
        self.rng_e2.copy_from_host(stream, &s2)?;
        self.rng_e3.copy_from_host(stream, &s3)?;
        let [s0, s1, s2, s3] = split(i_seeds);
        self.rng_i0.copy_from_host(stream, &s0)?;
        self.rng_i1.copy_from_host(stream, &s1)?;
        self.rng_i2.copy_from_host(stream, &s2)?;
        self.rng_i3.copy_from_host(stream, &s3)?;
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
    pub fn OLD_move_particles(
        efield:   &[Real],
        mut x:    DisjointSlice<Real>,
        mut vx:   DisjointSlice<Real>,
        n_active: &[u32],
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
            if i >= n_active[0] as usize {
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
    pub fn move_particles(
        efield:   &[Real],
        mut x:    DisjointSlice<Real>,
        mut vx:   DisjointSlice<Real>,
        n_active: &[u32],
        factor:   Real,
        dt:       Real,
    ) {
        if let Some((x_val, idx)) = x.get_mut_indexed() {
            let i = idx.get();
            if i >= n_active[0] as usize {
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

    #[kernel]
    pub fn get_density(
        x:             &[Real],
        density:       &[Real],
        n_active:      &[u32],
    ) {
        static mut LOCAL_DENSITY: SharedArray<Real, N_G> = SharedArray::UNINIT;

        let tid        = thread::threadIdx_x() as usize;
        let block_size = thread::blockDim_x()  as usize;

        let mut k = tid;
        while k < N_G {
            unsafe { LOCAL_DENSITY[k] = 0.0 as Real; }
            k += block_size;
        }
        thread::sync_threads();

        // Charge to shared histogram | Block-scope atomics
        let i = thread::index_1d().get();
        if i < n_active[0] as usize {
            let pos = x[i] * INV_DX as Real;
            let q   = pos as usize;
            let rem  = pos - q as Real;
            let c1   = (1.0 as Real - rem) * WEIGHT_FACTOR as Real;
            let c2   = rem * WEIGHT_FACTOR as Real;

            unsafe {
                let p: *mut SharedArray<Real, N_G> = &raw mut LOCAL_DENSITY;
                let local_base = (*p).as_mut_ptr() as *const BlockAtomicF32;

                (*local_base.add(q    )).fetch_add(c1, AtomicOrdering::Relaxed);
                (*local_base.add(q + 1)).fetch_add(c2, AtomicOrdering::Relaxed);
            };
        }
        thread::sync_threads();

        // Flush to global density | Device-scope atomics
        let global_density = density.as_ptr() as *const DeviceAtomicF32;
        let mut k = tid;
        while k < N_G {
            let mut val = unsafe { LOCAL_DENSITY[k] };
            if val != 0.0 as Real {
                if k == 0 || k == N_G - 1 {
                    val *= 2.0 as Real;
                }
                unsafe {
                    (*global_density.add(k)).fetch_add(val, AtomicOrdering::Relaxed);
                }
            }
            k += block_size;
        }
    }


    // Double-Prefix-Sum Poisson solver - f32 only.
    // Launch: 1 block, SCAN_BLOCK_SIZE=512 threads (≥ N_G=400).
    #[kernel]
    pub fn solve_poisson_scan_f32(
        e_density:  &[f32],
        i_density:  &[f32],
        mut pot:    DisjointSlice<f32>,
        mut efield: DisjointSlice<f32>,
        pot0:       f32,
    ) {
        static mut SCAN_SMEM: SharedArray<f32, SCAN_NUM_WARPS> = SharedArray::UNINIT;
        static mut H_S:       SharedArray<f32, N_G>            = SharedArray::UNINIT;
        static mut POT_S:     SharedArray<f32, N_G>            = SharedArray::UNINIT;
        static mut TOTAL_S:   SharedArray<f32, 1>              = SharedArray::UNINIT;

        const ALPHA_F: f32            = (-DX * DX / EPSILON0) as f32;
        const E_CHARGE_F:    f32      = E_CHARGE as f32;
        const INV_DX_F: f32           = INV_DX as f32;
        const HALF_DX_OVER_EPS_F: f32 = (DX / (2.0 * EPSILON0)) as f32;

        let tid   = thread::threadIdx_x() as usize;
        let block = this_thread_block();

        let g_i: f32 = if tid == 0 || tid >= N_G {
            0.0f32
        } else {
            let rho_i = E_CHARGE_F * (i_density[tid] - e_density[tid]);
            let f_i   = ALPHA_F * rho_i - if tid == 1 { pot0 } else { 0.0f32 };
            (tid as f32) * f_i
        };

        let s_i = block_scan::<f32, Sum, _>(&block, g_i, &raw mut SCAN_SMEM);

        let h_i: f32 = if tid == 0 || tid >= N_G {
            0.0f32
        } else {
            -s_i / (tid as f32 + 1.0f32)
        };

        if tid < N_G {
            unsafe { H_S[tid] = h_i; }
        }


        thread::sync_threads();


        let r_i: f32 = if tid == 0 || tid >= N_G - 1 {
            0.0f32
        } else {
            unsafe { H_S[tid] / (tid as f32) }
        };

        let big_r_i = block_scan::<f32, Sum, _>(&block, r_i, &raw mut SCAN_SMEM);

        if tid == N_G - 2 {
            unsafe { TOTAL_S[0] = big_r_i; }
        }


        thread::sync_threads();
        let total = unsafe { TOTAL_S[0] };


        let pot_i: f32 = if tid == 0 {
            pot0
        } else if tid < N_G - 1 {
            (tid as f32) * (total - big_r_i + r_i)
        } else {
            0.0f32 
        };

        if tid < N_G {
            unsafe { POT_S[tid] = pot_i; }
        }


        thread::sync_threads();


        // if tid < N_G {
        //     let pk = unsafe { POT_S[tid] };
        //     unsafe { *pot.get_unchecked_mut(tid) = pk; }

        //     let e_i = if tid == 0 {
        //         let rho0 = E_CHARGE_F * (i_density[0] - e_density[0]);
        //         let pk1  = unsafe { POT_S[1] };
        //         (pk - pk1) * INV_DX_F - rho0 * HALF_DX_OVER_EPS_F
        //     } else if tid == N_G - 1 {
        //         let rho_n = E_CHARGE_F * (i_density[N_G - 1] - e_density[N_G - 1]);
        //         let pkm1  = unsafe { POT_S[N_G - 2] };
        //         (pkm1 - pk) * INV_DX_F + rho_n * HALF_DX_OVER_EPS_F
        //     } else {
        //         let pkm1 = unsafe { POT_S[tid - 1] };
        //         let pkp1 = unsafe { POT_S[tid + 1] };
        //         0.5f32 * (pkm1 - pkp1) * INV_DX_F
        //     };
        //     unsafe { *efield.get_unchecked_mut(tid) = e_i; }
        // }
        if let Some((pot_elem, idx)) = pot.get_mut_indexed() {
            let pk = unsafe { POT_S[tid] };
            *pot_elem = pk;

            let e_i = if tid == 0 {
                let rho0 = E_CHARGE_F * (i_density[0] - e_density[0]);
                let pk1  = unsafe { POT_S[1] };
                (pk - pk1) * INV_DX_F - rho0 * HALF_DX_OVER_EPS_F
            } else if tid == N_G - 1 {
                let rho_n = E_CHARGE_F * (i_density[N_G - 1] - e_density[N_G - 1]);
                let pkm1  = unsafe { POT_S[N_G - 2] };
                (pkm1 - pk) * INV_DX_F + rho_n * HALF_DX_OVER_EPS_F
            } else {
                let pkm1 = unsafe { POT_S[tid - 1] };
                let pkp1 = unsafe { POT_S[tid + 1] };
                0.5f32 * (pkm1 - pkp1) * INV_DX_F
            };

            if let Some(efield_elem) = efield.get_mut(idx) {
                *efield_elem = e_i;
            }
        }
    }

    // per-block inclusive prefix scan over an
    // alive/dead flag (`block_scan`, fully parallel within a block in O(log N)
    // steps), plus a single `atomicAdd` per block to reserve a contiguous slot
    // range in the destination buffer. Each survivor writes to its globally unique
    // slot `base + inclusive - 1`.
    #[kernel]
    pub fn check_boundaries_compact(
        src_x:  &[Real],
        src_vx: &[Real],
        src_vy: &[Real],
        src_vz: &[Real],
        mut dst_x:  DisjointSlice<Real>,
        mut dst_vx: DisjointSlice<Real>,
        mut dst_vy: DisjointSlice<Real>,
        mut dst_vz: DisjointSlice<Real>,
        alive_counter: &[u32],   // global alive count (MUST BE zeroed before launch)
        n_active:      &[u32],
    ) {
        static mut SCAN_SMEM: SharedArray<u32, COMPACT_NUM_WARPS> = SharedArray::UNINIT;
        static mut BASE_S:    SharedArray<u32, 1>                 = SharedArray::UNINIT;

        let block = this_thread_block();
        let tid   = thread::threadIdx_x() as usize;
        let bdim  = thread::blockDim_x()  as usize;
        let i     = thread::index_1d().get();

        let mut flag: u32 = 0;
        if i < n_active[0] as usize {
            let xi = src_x[i];
            if xi >= 0.0 as Real && xi <= L as Real {
                flag = 1;
            }
        }

        let incl = block_scan::<u32, Sum, _>(&block, flag, &raw mut SCAN_SMEM);

        if tid == bdim - 1 {
            let g    = unsafe { DeviceAtomicU32::from_ptr(alive_counter.as_ptr() as *mut u32) };
            let base = g.fetch_add(incl, AtomicOrdering::Relaxed);
            unsafe { BASE_S[0] = base; }
        }

        thread::sync_threads();

        // each SURVIVING thread owns a distinct `slot`: prefix scan + per-block base
        if flag == 1 {
            let base = unsafe { BASE_S[0] };
            let slot = base as usize + incl as usize - 1; // inclusive scan -> subtract self
            unsafe {
                *dst_x.get_unchecked_mut(slot)  = src_x[i];
                *dst_vx.get_unchecked_mut(slot) = src_vx[i];
                *dst_vy.get_unchecked_mut(slot) = src_vy[i];
                *dst_vz.get_unchecked_mut(slot) = src_vz[i];
            }
        }
    }

    // fused move_particles + check_boundaries_compact kernel.
    #[kernel]
    pub fn move_and_compact(
        efield:   &[Real],
        src_x:    &[Real],
        src_vx:   &[Real],
        src_vy:   &[Real],
        src_vz:   &[Real],
        mut dst_x:  DisjointSlice<Real>,
        mut dst_vx: DisjointSlice<Real>,
        mut dst_vy: DisjointSlice<Real>,
        mut dst_vz: DisjointSlice<Real>,
        alive_counter: &[u32],   // global alive count (MUST be zeroed before launch)
        n_active:      &[u32],
        factor:   Real,
        dt:       Real,
    ) {
        static mut SCAN_SMEM: SharedArray<u32, COMPACT_NUM_WARPS> = SharedArray::UNINIT;
        static mut BASE_S:    SharedArray<u32, 1>                 = SharedArray::UNINIT;

        let block = this_thread_block();
        let tid   = thread::threadIdx_x() as usize;
        let bdim  = thread::blockDim_x()  as usize;
        let i     = thread::index_1d().get();

        let mut flag: u32 = 0;
        let mut new_x:  Real = 0.0;
        let mut new_vx: Real = 0.0;

        if i < n_active[0] as usize {
            let xi  = src_x[i];
            let vxi = src_vx[i];

            let pos = xi * INV_DX as Real;
            let p   = pos as usize;
            let c2  = pos - p as Real;
            let e_x = (1.0 as Real - c2) * efield[p]
                    + c2 * efield[p + 1];

            new_vx = vxi + factor * e_x;
            new_x  = xi  + new_vx * dt;

            if new_x >= 0.0 as Real && new_x <= L as Real {
                flag = 1;
            }
        }

        let incl = block_scan::<u32, Sum, _>(&block, flag, &raw mut SCAN_SMEM);

        if tid == bdim - 1 {
            let g    = unsafe { DeviceAtomicU32::from_ptr(alive_counter.as_ptr() as *mut u32) };
            let base = g.fetch_add(incl, AtomicOrdering::Relaxed);
            unsafe { BASE_S[0] = base; }
        }
        thread::sync_threads();

        if flag == 1 {
            let base = unsafe { BASE_S[0] };
            let slot = base as usize + incl as usize - 1;
            unsafe {
                *dst_x.get_unchecked_mut(slot)  = new_x;
                *dst_vx.get_unchecked_mut(slot) = new_vx;
                *dst_vy.get_unchecked_mut(slot) = src_vy[i];
                *dst_vz.get_unchecked_mut(slot) = src_vz[i];
            }
        }
    }

    /// testing: flexible Double-Prefix-Sum Poisson solver for convergence testing
    /// solves ψ[i-1] - 2ψ[i] + ψ[i+1] = f[i] where f[i] = dx²·source[i].
    /// outputs only potential (no E-field) since convergence test needs only ψ.
    #[kernel]
    pub fn solve_poisson_dps_flexible(
        source:    &[f32],              // source term s(x_i), length = n
        mut pot:   DisjointSlice<f32>,  // output potential, length = n
        n:         u32,                 // number of grid points (including boundaries)
        psi_left:  f32,                 // ψ(0) boundary condition
        psi_right: f32,                 // ψ(N-1) boundary condition
        dx:        f32,                 // grid spacing
    ) {
        static mut SCAN_SMEM: SharedArray<f32, DPS_TEST_WARPS>   = SharedArray::UNINIT;
        static mut H_S:       SharedArray<f32, DPS_TEST_MAX_N>   = SharedArray::UNINIT;
        static mut TOTAL_S:   SharedArray<f32, 1>                = SharedArray::UNINIT;

        let tid   = thread::threadIdx_x() as usize;
        let nn    = n as usize;
        let block = this_thread_block();
        let dx2   = dx * dx;

        let g_i: f32 = if tid == 0 || tid >= nn {
            0.0f32
        } else {
            let mut f_i = dx2 * source[tid];
            if tid == 1        { f_i -= psi_left; }
            if tid == nn - 2   { f_i -= psi_right; }
            (tid as f32) * f_i
        };

        let s_i = block_scan::<f32, Sum, _>(&block, g_i, &raw mut SCAN_SMEM);

        let h_i: f32 = if tid == 0 || tid >= nn {
            0.0f32
        } else {
            -s_i / (tid as f32 + 1.0f32)
        };

        if tid < nn {
            unsafe { H_S[tid] = h_i; }
        }
        thread::sync_threads();

        let r_i: f32 = if tid == 0 || tid >= nn - 1 {
            0.0f32
        } else {
            unsafe { H_S[tid] / (tid as f32) }
        };

        let big_r_i = block_scan::<f32, Sum, _>(&block, r_i, &raw mut SCAN_SMEM);

        if tid == nn - 2 {
            unsafe { TOTAL_S[0] = big_r_i; }
        }
        thread::sync_threads();
        let total = unsafe { TOTAL_S[0] };

        let pot_i: f32 = if tid == 0 {
            psi_left
        } else if tid < nn - 1 {
            (tid as f32) * (total - big_r_i + r_i)
        } else if tid == nn - 1 {
            psi_right
        } else {
            0.0f32
        };

        if let Some((pot_elem, _idx)) = pot.get_mut_indexed() {
            if tid < nn {
                *pot_elem = pot_i;
            }
        }
    }

    #[inline(always)]                                                                                                                                                                
    fn ptx_abs(x: Real) -> Real {                                                                                                                                                    
        let result: Real;
        unsafe {
            ptx_asm!(
                "abs.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }


    #[inline(always)]
    fn ptx_sqrt(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "sqrt.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    #[inline(always)]
    fn ptx_sin(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "sin.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    #[inline(always)]
    fn ptx_cos(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "cos.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    /// 2^x
    #[inline(always)]
    fn ptx_ex2(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "ex2.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    /// log2(x)
    #[inline(always)]
    fn ptx_lg2(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "lg2.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    /// 1/x
    #[inline(always)]
    fn ptx_rcp(x: Real) -> Real {
        let result: Real;
        unsafe {
            ptx_asm!(
                "rcp.approx.f32 %0, %1;",
                out("=f") result,
                in("f") x,
                options(register_only),
            );
        }
        result
    }

    /// 2^(x * log2(e))
    #[inline(always)]
    fn ptx_exp(x: Real) -> Real {
        ptx_ex2(x * LOG2_E)
    }

    /// sin(x) / cos(x)
    #[inline(always)]
    fn ptx_tan(x: Real) -> Real {
        let s = ptx_sin(x);
        let c = ptx_cos(x);
        s * ptx_rcp(c)
    }

    /// for |x| > 1, atan(x) = pi/2 - atan(1/x)
    #[inline(always)]
    fn ptx_atan(x: Real) -> Real {
        let ax = if x < 0.0 { -x } else { x };
        let swap = ax > 1.0;
        let z = if swap { ptx_rcp(ax) } else { ax };

        let z2 = z * z;
        let mut r = -0.0464964749;
        r = r * z2 + 0.15931422;
        r = r * z2 - 0.327622764;
        r = r * z2 + 0.999847695;
        r = r * z;

        if swap {
            r = (0.5 * PI as Real) - r;
        }
        if x < 0.0 { -r } else { r }
    }

    #[inline(always)]
    fn ptx_atan2(y: Real, x: Real) -> Real {
        let ax = if x < 0.0 { -x } else { x };
        let ay = if y < 0.0 { -y } else { y };

        if ax < 1e-30 && ay < 1e-30 {
            return 0.0;
        }

        let a: Real;
        let swap = ay > ax;
        if swap {
            a = ptx_atan(ax * ptx_rcp(ay));
        } else {
            a = ptx_atan(ay * ptx_rcp(ax));
        }

        let r = if swap { (0.5 * PI as Real) - a } else { a };
        let r = if x < 0.0 { PI as Real - r } else { r };
        if y < 0.0 { -r } else { r }
    }

    /// atan2(sqrt(1 - x*x), x)
    #[inline(always)]
    fn ptx_acos(x: Real) -> Real {
        let clamped = if x > 1.0 { 1.0 } else if x < -1.0 { -1.0 } else { x };
        let s = ptx_sqrt(ptx_abs(1.0 - clamped * clamped));
        ptx_atan2(s, clamped)
    }

    #[inline(always)]
    fn ptx_max(x: Real, y: Real) -> Real {
        let mut result;
        unsafe {
            ptx_asm!(
                "max.f32 %0, %1, %2;",
                out("=f") result,
                in("f") x,
                in("f") y,
                options(register_only),
            );
        }
        result
    }

    fn rotl32(x: u32, k: u32) -> u32 {
        (x << k) | (x >> (32 - k))
    }

    fn xoshiro128p_next(s0: u32, s1: u32, s2: u32, s3: u32) -> (u32, u32, u32, u32, u32) {
        let result = s0 + s3;
        let t = s1 << 9;
        let s2 = s2 ^ s0;
        let s3 = s3 ^ s1;
        let s1 = s1 ^ s2;
        let s0 = s0 ^ s3;
        let s2 = s2 ^ t;
        let s3 = rotl32(s3, 11);
        (result, s0, s1, s2, s3)
    }

    fn u32_to_real(x: u32) -> Real {
        (f32::from_bits(0x3F80_0000u32 | (x >> 9)) - 1.0) as Real
    }

    fn rng_next_f32(i: usize, rng0: &mut DisjointSlice<u32>, rng1: &mut DisjointSlice<u32>, rng2: &mut DisjointSlice<u32>, rng3: &mut DisjointSlice<u32>) -> Real {
        let s0 = unsafe { *rng0.get_unchecked_mut(i) };
        let s1 = unsafe { *rng1.get_unchecked_mut(i) };
        let s2 = unsafe { *rng2.get_unchecked_mut(i) };
        let s3 = unsafe { *rng3.get_unchecked_mut(i) };
        let (result, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        unsafe {
            *rng0.get_unchecked_mut(i) = ns0;
            *rng1.get_unchecked_mut(i) = ns1;
            *rng2.get_unchecked_mut(i) = ns2;
            *rng3.get_unchecked_mut(i) = ns3;
        }
        u32_to_real(result)
    }

    fn rng_next_three_normal(
        i: usize, 
        sigma: Real, // standard deviation of the normal distribution
        rng0: &mut DisjointSlice<u32>, 
        rng1: &mut DisjointSlice<u32>, 
        rng2: &mut DisjointSlice<u32>, 
        rng3: &mut DisjointSlice<u32>
    ) -> (Real, Real, Real) {
        
        let mut s0 = unsafe { *rng0.get_unchecked_mut(i) };
        let mut s1 = unsafe { *rng1.get_unchecked_mut(i) };
        let mut s2 = unsafe { *rng2.get_unchecked_mut(i) };
        let mut s3 = unsafe { *rng3.get_unchecked_mut(i) };

        let (r1, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        s0 = ns0; s1 = ns1; s2 = ns2; s3 = ns3;

        let (r2, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        s0 = ns0; s1 = ns1; s2 = ns2; s3 = ns3;

        let (r3, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        s0 = ns0; s1 = ns1; s2 = ns2; s3 = ns3;

        // fourth random for Box-Muller, but not used
        let (r4, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        
        unsafe {
            *rng0.get_unchecked_mut(i) = ns0;
            *rng1.get_unchecked_mut(i) = ns1;
            *rng2.get_unchecked_mut(i) = ns2;
            *rng3.get_unchecked_mut(i) = ns3;
        }

        let f1 = u32_to_real(r1);
        let f2 = u32_to_real(r2);
        let f3 = u32_to_real(r3);
        let f4 = u32_to_real(r4);
        let u1 = ptx_max(f1, 1e-9 as Real);
        let u2 = ptx_max(f2, 1e-9 as Real);
        let u3 = ptx_max(f3, 1e-9 as Real);
        let u4 = ptx_max(f4, 1e-9 as Real);

        let rad1 = ptx_sqrt(-2.0 as Real * ptx_lg2(u1) * (1.0 / LOG2_E));
        let angle1 = TWO_PI as Real * u2;
        
        let z0 = rad1 * ptx_cos(angle1);
        let z1 = rad1 * ptx_sin(angle1);

        let rad2 = ptx_sqrt(-2.0 as Real * ptx_lg2(u3) * (1.0 / LOG2_E));
        let angle2 = TWO_PI as Real * u4;
        
        let z2 = rad2 * ptx_cos(angle2);

        (z0 * sigma, z1 * sigma, z2 * sigma)
    }

    #[kernel]
    pub fn check_collisions_e(total_cs_e: &[Real], cs: &[Real], active_e: &[u32], 
                        mut x: DisjointSlice<Real>, mut vx: DisjointSlice<Real>, mut vy: DisjointSlice<Real>, mut vz: DisjointSlice<Real>, 
                        mut rng0: DisjointSlice<u32>, mut rng1: DisjointSlice<u32>, mut rng2: DisjointSlice<u32>, mut rng3: DisjointSlice<u32>,
                        mut i_x: DisjointSlice<Real>, mut i_vx: DisjointSlice<Real>, mut i_vy: DisjointSlice<Real>, mut i_vz: DisjointSlice<Real>,
                        alive_e: &[u32], alive_i: &[u32]) {
        let i = thread::index_1d().get();
        if i >= active_e[0] as usize {
            return;
        } 

        let v2 = unsafe { *vx.get_unchecked_mut(i) * *vx.get_unchecked_mut(i) + *vy.get_unchecked_mut(i) * *vy.get_unchecked_mut(i) + *vz.get_unchecked_mut(i) * *vz.get_unchecked_mut(i) };
        let energy: Real = HALF_E_MASS_OVER_E_CHARGE * v2; // EV_TO_J
        let c1 = (energy / (DE_CS as Real) + 0.5) as usize;
        let c2 = CS_RANGES - 1;

        let energy_index = c1.min(c2);
        let nu: Real = if PRECOMPUTE_COLLISION_FREQ {
            total_cs_e[energy_index]
        } else {
            let velocity: Real = ptx_sqrt(v2);
            total_cs_e[energy_index] * velocity
        };

        let rand_val = rng_next_f32(i, &mut rng0, &mut rng1, &mut rng2, &mut rng3);
        
        let p_coll: Real = 1.0 - ptx_exp(-nu * DT_E as Real);
        if rand_val < p_coll {
            collision_e(cs, &mut x, &mut vx, &mut vy, &mut vz, i, energy_index, &mut rng0, &mut rng1, &mut rng2, &mut rng3,
                       &mut i_x, &mut i_vx, &mut i_vy, &mut i_vz, alive_e, alive_i);
        }
        
    }

    // TODO - options(may_diverge) sprawdź dla inline ptx.

    pub fn collision_e(cs: &[Real], x: &mut DisjointSlice<Real>, vx: &mut DisjointSlice<Real>, vy: &mut DisjointSlice<Real>, vz: &mut DisjointSlice<Real>, i: usize, 
                    energy_index: usize, rng0: &mut DisjointSlice<u32>, rng1: &mut DisjointSlice<u32>, rng2: &mut DisjointSlice<u32>, rng3: &mut DisjointSlice<u32>,
                    i_x: &mut DisjointSlice<Real>, i_vx: &mut DisjointSlice<Real>, i_vy: &mut DisjointSlice<Real>, i_vz: &mut DisjointSlice<Real>, alive_e: &[u32], alive_i: &[u32]) {
        let mut gx: Real = unsafe { *vx.get_unchecked_mut(i) };
        let mut gy: Real = unsafe { *vy.get_unchecked_mut(i) };
        let mut gz: Real = unsafe { *vz.get_unchecked_mut(i) };
        let mut g: Real = ptx_sqrt(gx * gx + gy * gy + gz * gz);
        let wx: Real = F1 * unsafe { *vx.get_unchecked_mut(i) };
        let wy: Real = F1 * unsafe { *vy.get_unchecked_mut(i) };
        let wz: Real = F1 * unsafe { *vz.get_unchecked_mut(i) };

        // Cross-section lookup using energy_index (computed in check_collisions_e)
        let t0: Real = cs[E_ELA * CS_RANGES + energy_index];
        let t1: Real = t0 + cs[E_EXC * CS_RANGES + energy_index];
        let t2: Real = t1 + cs[E_ION * CS_RANGES + energy_index];

        let phi: Real;
        let theta: Real;
        if gx == 0.0 as Real {
            theta = 0.5 * PI as Real;
        } else {
            theta = ptx_atan2(ptx_sqrt(gy * gy + gz * gz), gx);
        }

        if gy == 0.0 as Real {
            if gz >= 0.0 as Real { phi = 0.5 * PI as Real; }
            else { phi = -0.5 * PI as Real; }
        } else {
            phi = ptx_atan2(gz, gy);
        }

        let chi: Real;
        let eta: Real;
        let mut sc: Real;
        let mut cc: Real;
        let mut se: Real;
        let mut ce: Real;
        let st: Real = ptx_sin(theta);
        let ct: Real = ptx_cos(theta);
        let sp: Real = ptx_sin(phi);
        let cp: Real = ptx_cos(phi);

        let rnd = rng_next_f32(i, rng0, rng1, rng2, rng3);
        if rnd < t0 / t2 {  // elastic scattering
            chi = ptx_acos(1.0 - 2.0 * rng_next_f32(i, rng0, rng1, rng2, rng3));
            eta = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);
        } else if rnd < t1 / t2 {  // excitation
            let mut energy = 0.5 * E_MASS as Real * g * g;
            energy = ptx_abs(energy - E_EXC_TH as Real * E_CHARGE as Real);
            g = ptx_sqrt(2.0 as Real * energy / E_MASS as Real);
            chi = ptx_acos(1.0 - 2.0 * rng_next_f32(i, rng0, rng1, rng2, rng3));
            eta = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);
        } else {  // ionization
            let mut energy = 0.5 * E_MASS as Real * g * g;
            energy = ptx_abs(energy - E_ION_TH as Real * E_CHARGE as Real);
            let e_new = 10.0 as Real * ptx_tan(rng_next_f32(i, rng0, rng1, rng2, rng3) * ptx_atan(energy / E_CHARGE as Real/20.0)) * E_CHARGE as Real;
            let e_orig = ptx_abs(energy - e_new);
            g = ptx_sqrt(2.0 as Real * e_orig / E_MASS as Real);
            let g_new: Real = ptx_sqrt(2.0 as Real * e_new / E_MASS as Real);
            chi = ptx_acos(ptx_sqrt(e_orig / energy));
            let chi_new: Real = ptx_acos(ptx_sqrt(e_new / energy));
            eta = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);
            let eta_new: Real = eta + PI as Real;
            sc = ptx_sin(chi_new);
            cc = ptx_cos(chi_new);
            se = ptx_sin(eta_new);
            ce = ptx_cos(eta_new);
            gx = g_new * (ct * cc - st * sc * ce);
            gy = g_new * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
            gz = g_new * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
            
            let n_alive = unsafe { &*(alive_e.as_ptr() as *const DeviceAtomicU32) };
            let idx = n_alive.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe {
                *x.get_unchecked_mut(idx as usize) = *x.get_unchecked_mut(i);
                *vx.get_unchecked_mut(idx as usize) = wx + F2 * gx;
                *vy.get_unchecked_mut(idx as usize) = wy + F2 * gy;
                *vz.get_unchecked_mut(idx as usize) = wz + F2 * gz;
            }
            
            let n_ions_alive = unsafe { &*(alive_i.as_ptr() as *const DeviceAtomicU32) };
            let ion_idx = n_ions_alive.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe {
                *i_x.get_unchecked_mut(ion_idx as usize) = *x.get_unchecked_mut(i);
                *i_vx.get_unchecked_mut(ion_idx as usize) = 0.0 as Real; // TODO - properly initialize ion v
                *i_vy.get_unchecked_mut(ion_idx as usize) = 0.0 as Real;
                *i_vz.get_unchecked_mut(ion_idx as usize) = 0.0 as Real;
            }
        }

        sc = ptx_sin(chi);
        cc = ptx_cos(chi);
        se = ptx_sin(eta);
        ce = ptx_cos(eta);
        gx = g * (ct * cc - st * sc * ce);
        gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

        unsafe {
            *vx.get_unchecked_mut(i) = wx + F2 * gx;
            *vy.get_unchecked_mut(i) = wy + F2 * gy;
            *vz.get_unchecked_mut(i) = wz + F2 * gz;
        }
    }

    #[kernel]
    pub fn check_collisions_i(
        total_cs_i: &[Real],
        cs: &[Real],
        active_i: &[u32],
        mut vx: DisjointSlice<Real>,
        mut vy: DisjointSlice<Real>,
        mut vz: DisjointSlice<Real>,
        mut rng0: DisjointSlice<u32>,
        mut rng1: DisjointSlice<u32>,
        mut rng2: DisjointSlice<u32>,
        mut rng3: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= active_i[0] as usize {
            return;
        }

        let (vxa, vya, vza) = rng_next_three_normal(i, NORMAL_RANGE, &mut rng0, &mut rng1, &mut rng2, &mut rng3);

        let gx = unsafe { *vx.get_unchecked_mut(i) } - vxa;
        let gy = unsafe { *vy.get_unchecked_mut(i) } - vya;
        let gz = unsafe { *vz.get_unchecked_mut(i) } - vza;
        let g2 = gx * gx + gy * gy + gz * gz;

        let energy: Real = HALF_MU_ARAR_OVER_E_CHARGE * g2;
        let c1 = (energy / (DE_CS as Real) + 0.5) as usize;
        let c2 = CS_RANGES - 1;
        let energy_index = c1.min(c2);

        let nu: Real = if PRECOMPUTE_COLLISION_FREQ { // TODO - can be removed if we do the choice
            total_cs_i[energy_index]
        } else {
            let g = ptx_sqrt(g2);
            total_cs_i[energy_index] * g
        };

        let rand_val = rng_next_f32(i, &mut rng0, &mut rng1, &mut rng2, &mut rng3);
        let p_coll: Real = 1.0 - ptx_exp(-nu * DT_I as Real);
        if rand_val < p_coll {
            collision_i(cs, &mut vx, &mut vy, &mut vz, i,
                        vxa, vya, vza, energy_index,
                        &mut rng0, &mut rng1, &mut rng2, &mut rng3);
        }
    }

    pub fn collision_i(
        cs: &[Real],
        vx: &mut DisjointSlice<Real>,
        vy: &mut DisjointSlice<Real>,
        vz: &mut DisjointSlice<Real>,
        i: usize,
        vxa: Real,
        vya: Real,
        vza: Real,
        energy_index: usize,
        rng0: &mut DisjointSlice<u32>,
        rng1: &mut DisjointSlice<u32>,
        rng2: &mut DisjointSlice<u32>,
        rng3: &mut DisjointSlice<u32>,
    ) {
        let t0: Real = cs[I_ISO * CS_RANGES + energy_index];
        let t1: Real = t0 + cs[I_BACK * CS_RANGES + energy_index];

        let phi: Real;
        let theta: Real;
        let chi: Real;

        let mut gx: Real = unsafe { *vx.get_unchecked_mut(i) } - vxa;
        let mut gy: Real = unsafe { *vy.get_unchecked_mut(i) } - vya;
        let mut gz: Real = unsafe { *vz.get_unchecked_mut(i) } - vza;
        let g: Real = ptx_sqrt(gx * gx + gy * gy + gz * gz);
        let wx: Real = 0.5 * (unsafe { *vx.get_unchecked_mut(i) } + vxa);
        let wy: Real = 0.5 * (unsafe { *vy.get_unchecked_mut(i) } + vya);
        let wz: Real = 0.5 * (unsafe { *vz.get_unchecked_mut(i) } + vza);

        if gx == 0.0 as Real {
            theta = 0.5 * PI as Real;
        } else {
            theta = ptx_atan2(ptx_sqrt(gy * gy + gz * gz), gx);
        }
        if gy == 0.0 as Real {
            if gz >= 0.0 as Real { phi = 0.5 * PI as Real; }
            else { phi = -0.5 * PI as Real; }
        } else {
            phi = ptx_atan2(gz, gy);
        }

        let rnd = rng_next_f32(i, rng0, rng1, rng2, rng3);
        if rnd < t0 / t1 {
            chi = ptx_acos(1.0 - 2.0 * rng_next_f32(i, rng0, rng1, rng2, rng3));
        } else {
            chi = PI as Real;
        }
        let eta: Real = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);

        let sc: Real = ptx_sin(chi);
        let cc: Real = ptx_cos(chi);
        let se: Real = ptx_sin(eta);
        let ce: Real = ptx_cos(eta);
        let st: Real = ptx_sin(theta);
        let ct: Real = ptx_cos(theta);
        let sp: Real = ptx_sin(phi);
        let cp: Real = ptx_cos(phi);

        gx = g * (ct * cc - st * sc * ce);
        gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

        unsafe {
            *vx.get_unchecked_mut(i) = wx + 0.5 * gx;
            *vy.get_unchecked_mut(i) = wy + 0.5 * gy;
            *vz.get_unchecked_mut(i) = wz + 0.5 * gz;
        }
    }
}

// Host-side initialization helpers

fn init_particles(n: usize) -> ParticlesSoA {
    let mut rng = rand::rng();
    let sigma_v = (K_BOLTZMANN * TEMPERATURE / AR_MASS).sqrt();
    let normal  = Normal::new(0.0f64, sigma_v).unwrap();

    let mut particles = ParticlesSoA::with_capacity(MAX_PARTICLES);

    for i in 0..n {
        particles.x[i]  = (rng.random::<f64>() * L) as Real;
        particles.vx[i] = rng.sample(normal) as Real;
        particles.vy[i] = rng.sample(normal) as Real;
        particles.vz[i] = rng.sample(normal) as Real;
    }

    particles
}

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

/// generate xoshiro128+ seed state for all particle slots.
fn xoshiro128_seed_streams(master_seed: [u32; 4], n: usize) -> Vec<[u32; 4]> {
    debug_assert!(master_seed != [0, 0, 0, 0], "state should not be everywhere 0");

    const JUMP: [u32; 4] = [0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b];

    fn rotl32(x: u32, k: u32) -> u32 { (x << k) | (x >> (32 - k)) }
    fn next(s: &mut [u32; 4]) -> u32 {
        let result = s[0] + s[3];

        let t = s[1] << 9;

        s[2] ^= s[0]; 
        s[3] ^= s[1];
        s[1] ^= s[2]; 
        s[0] ^= s[3];

        s[2] ^= t;    
        
        s[3] = rotl32(s[3], 11);

        result
    }
    fn jump(s: &mut [u32; 4]) {
        let mut acc = [0u32; 4];
        for &word in JUMP.iter() {
            for b in 0..32 {
                if word & (1u32 << b) != 0 {
                    acc[0] ^= s[0];
                    acc[1] ^= s[1];
                    acc[2] ^= s[2];
                    acc[3] ^= s[3];
                }
                next(s);
            }
        }
        *s = acc;
    }

    let mut state = master_seed;
    let mut streams = Vec::with_capacity(n);
    for _ in 0..n {
        streams.push(state);
        jump(&mut state);
    }
    streams
}


enum ParticleSpecies {
    Electrons = 0,
    Ions      = 1,
}

impl fmt::Display for ParticleSpecies {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParticleSpecies::Electrons => write!(f, "electrons"),
            ParticleSpecies::Ions      => write!(f, "ions"),
        }
    }
}


use std::fs::{self, File};
use std::io::Write;
use chrono::{DateTime, Utc};
use chrono_tz::Europe::Warsaw;
use chrono_tz::Tz;

fn save_particle_data(particles: &ParticlesSoA, amount: usize, step: usize, species: ParticleSpecies, tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();

    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("Unable to create directory");
    let filename = format!("{}/{:04}_{}_{}.csv", dir_path, step, time_stamp, species);

    let mut file = File::create(&filename).expect("Unable to create file");
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "x,vx,vy,vz").expect("Unable to write header");
    
    for i in 0..amount {
        writeln!(
            writer,
            "{},{},{},{}",
            particles.x[i], particles.vx[i], particles.vy[i], particles.vz[i]
        )
        .expect("Unable to write particle data");
    }
}

const CHECKPOINT_CYCLES: usize = 10;

fn save_particle_growth_data(n_e: Vec<u32>, n_i: Vec<u32>, tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();

    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("Unable to create directory");
    let filename = format!("{}/particle_growth_{}.csv", dir_path, time_stamp);

    let mut file = File::create(&filename).expect("Unable to create file");
    writeln!(file, "step,n_e,n_i").expect("Unable to write header");

    for (step, (&n_e_val, &n_i_val)) in n_e.iter().zip(n_i.iter()).enumerate() {
        writeln!(file, "{},{},{}", step*CHECKPOINT_CYCLES, n_e_val, n_i_val).expect("Unable to write particle growth data");
    }
}




fn main() {
    // perform_tests();

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
    let electrons_host = init_particles(N_INIT);
    let ions_host      = init_particles(N_INIT);

    // 4. Allocate all GPU buffers
    let mut gpu = GpuSimState::allocate(&stream)
        .expect("Failed to allocate GPU memory");
    println!(">> eduPIC-GPU: GPU memory allocated (~{:.1} MB)",
        (MAX_PARTICLES * PARTICLE_COMPS * SIZEOF_REAL * N_SPECIES   // particles (e + i)
        + N_CS * CS_RANGES * SIZEOF_REAL                            // cross sections
        + CS_RANGES * SIZEOF_REAL * N_SPECIES                       // sigma_tot_e + sigma_tot_i
        + N_G * N_GRID_ARRAYS * SIZEOF_REAL                         // grid arrays
        + MAX_PARTICLES * 4 * 2                                     // RNG state (always u32)
        ) as f64 / BYTES_PER_MB
    );

    // 5. Upload data to GPU (one-time PCIe transfer)
    gpu.upload_electrons(&stream, &electrons_host, N_INIT as u32)
        .expect("Failed to upload electrons");
    gpu.upload_ions(&stream, &ions_host, N_INIT as u32)
        .expect("Failed to upload ions");
    gpu.upload_cross_sections(&stream, &cs_flat, &sigma_tot_e, &sigma_tot_i)
        .expect("Failed to upload cross-sections");

    let e_seeds = xoshiro128_seed_streams([0x1234_5678, 0x1111_2222, 0x2222_3333, 0x3333_4444], MAX_PARTICLES);
    let i_seeds = xoshiro128_seed_streams([0x4444_5555, 0x5555_6666, 0x6666_7777, 0x7777_8888], MAX_PARTICLES);
    gpu.upload_rng_state(&stream, &e_seeds, &i_seeds)
        .expect("Failed to upload RNG state");

    println!(">> eduPIC-GPU: data uploaded to GPU");

    // 6. Launch configs
    let cfg = LaunchConfig::for_num_elems(MAX_PARTICLES_U32);
    let poisson_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (SCAN_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut h_counter_e = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
    let mut h_counter_i = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();

    // 7. GPU simulation loop
    println!(">> eduPIC-GPU: running {} cycles x{} steps...", num_cycles, N_T);
    let module = kernels::load(&ctx).expect("Failed to load CUDA module");

    let mut n_e: u32 = N_INIT as u32;
    let mut n_i: u32 = N_INIT as u32;
    let mut n_e_history: Vec<u32> = Vec::with_capacity(num_cycles / CHECKPOINT_CYCLES + 1);
    let mut n_i_history: Vec<u32> = Vec::with_capacity(num_cycles / CHECKPOINT_CYCLES + 1);
    n_e_history.push(n_e);
    n_i_history.push(n_i);

    let cfg_e = cfg;
    let cfg_i = cfg;

    for cycle in 0..num_cycles {
        for t in 0..N_T {
            gpu.e_density.zero_async(&stream).expect("Failed to zero e_density");
            module.get_density(&stream, cfg_e,
                &gpu.e_x, &gpu.e_density, &gpu.n_electrons,
            ).expect("get_density (electrons) failed");

            if t % N_SUB == 0 {
                gpu.i_density.zero_async(&stream).expect("Failed to zero i_density");
                module.get_density(&stream, cfg_i,
                    &gpu.i_x, &gpu.i_density, &gpu.n_ions,
                ).expect("get_density (ions) failed");
            }

            let pot0 = (VOLTAGE * ((t as f64 / N_T as f64) * TWO_PI ).cos() as f64) as f32;
            module.solve_poisson_scan_f32(&stream, poisson_cfg,
                &gpu.e_density, &gpu.i_density, &mut gpu.pot, &mut gpu.efield, pot0,
            ).expect("solve_poisson failed");

            gpu.alive_counter.zero_async(&stream).expect("failed to zero alive_counter");
            module.move_and_compact(&stream, cfg_e, &gpu.efield,
                &gpu.e_x, &gpu.e_vx, &gpu.e_vy, &gpu.e_vz,
                &mut gpu.tmp_x, &mut gpu.tmp_vx, &mut gpu.tmp_vy, &mut gpu.tmp_vz,
                &gpu.alive_counter, &gpu.n_electrons, FACTOR_E as Real, DT_E as Real
            ).expect("move_and_compact (electrons) failed");
            std::mem::swap(&mut gpu.e_x,  &mut gpu.tmp_x);
            std::mem::swap(&mut gpu.e_vx, &mut gpu.tmp_vx);
            std::mem::swap(&mut gpu.e_vy, &mut gpu.tmp_vy);
            std::mem::swap(&mut gpu.e_vz, &mut gpu.tmp_vz);
            gpu.n_electrons.copy_from_device_async(&gpu.alive_counter, &stream).expect("failed to copy alive_counter to n_electrons");

            if t % N_SUB == 0 {
                gpu.alive_counter.zero_async(&stream).expect("failed to zero alive_counter");
                module.move_and_compact(&stream, cfg_i, &gpu.efield,
                    &gpu.i_x, &gpu.i_vx, &gpu.i_vy, &gpu.i_vz,
                    &mut gpu.tmp_x, &mut gpu.tmp_vx, &mut gpu.tmp_vy, &mut gpu.tmp_vz,
                    &gpu.alive_counter, &gpu.n_ions,  FACTOR_I as Real, DT_I as Real
                ).expect("move_and_compact (ions) failed");
                std::mem::swap(&mut gpu.i_x,  &mut gpu.tmp_x);
                std::mem::swap(&mut gpu.i_vx, &mut gpu.tmp_vx);
                std::mem::swap(&mut gpu.i_vy, &mut gpu.tmp_vy);
                std::mem::swap(&mut gpu.i_vz, &mut gpu.tmp_vz);
                gpu.n_ions.copy_from_device_async(&gpu.alive_counter, &stream).expect("failed to copy alive_counter to n_ions");
            }

            gpu.alive_counter.copy_from_device_async(&gpu.n_electrons, &stream).expect("failed to copy n_electrons to alive_counter");
            module.check_collisions_e(&stream, cfg_e,
                &gpu.sigma_tot_e, &gpu.cs, &gpu.n_electrons,
                &mut gpu.e_x, &mut gpu.e_vx, &mut gpu.e_vy, &mut gpu.e_vz,
                &mut gpu.rng_e0, &mut gpu.rng_e1, &mut gpu.rng_e2, &mut gpu.rng_e3,
                &mut gpu.i_x, &mut gpu.i_vx, &mut gpu.i_vy, &mut gpu.i_vz,
                &gpu.alive_counter, &gpu.n_ions,
            ).expect("check_collisions_e failed");
            gpu.n_electrons.copy_from_device_async(&gpu.alive_counter, &stream).expect("failed to copy alive_counter to n_electrons");


            if t % N_SUB == 0 {
                module.check_collisions_i(&stream, cfg_i,
                    &gpu.sigma_tot_i, &gpu.cs, &gpu.n_ions,
                    &mut gpu.i_vx, &mut gpu.i_vy, &mut gpu.i_vz,
                    &mut gpu.rng_i0, &mut gpu.rng_i1, &mut gpu.rng_i2, &mut gpu.rng_i3,
                ).expect("check_collisions_i failed");
            }
        }

        if (cycle + 1) % CHECKPOINT_CYCLES == 0 {
            unsafe { gpu.n_electrons.copy_to_pinned_host_async(&stream, &mut h_counter_e) }
                .expect("copy_to_pinned_host_async n_electrons checkpoint failed");
            unsafe { gpu.n_ions.copy_to_pinned_host_async(&stream, &mut h_counter_i) }
                .expect("copy_to_pinned_host_async n_ions checkpoint failed");
                
            stream.synchronize().unwrap();

            println!("   checkpoint at cycle {}: n_e={}, n_i={}", cycle + 1, h_counter_e[0], h_counter_i[0]);
            n_e_history.push(h_counter_e[0]);
            n_i_history.push(h_counter_i[0]);
        }
    }

    // 8. Synchronize and download results
    ctx.synchronize().expect("CUDA synchronization failed");

    let (_electrons_result, n_e_final) = gpu.download_electrons(&stream)
        .expect("Failed to download electrons");
    let (_ions_result, n_i_final) = gpu.download_ions(&stream)
        .expect("Failed to download ions");

    let elapsed = start.elapsed().as_secs_f64();
    println!(">> eduPIC-GPU: simulation complete in {:.3} s", elapsed);
    println!(">> eduPIC-GPU: final particles: {} electrons, {} ions", n_e_final, n_i_final);

    let tsmp = chrono::Utc::now().with_timezone(&Warsaw);
    save_particle_data(&_electrons_result, n_e_final as usize, num_cycles, ParticleSpecies::Electrons, tsmp);
    save_particle_data(&_ions_result, n_i_final as usize, num_cycles, ParticleSpecies::Ions, tsmp);
    save_particle_growth_data(n_e_history, n_i_history, tsmp);
}

// tests

fn perform_tests() {
    test_move_particles_analytic();
    test_move_particles_edge_cases();
    test_move_particles();
    test_dps_convergence();
    test_dps_convergence_gpu();
    bench_shmem_vs_no_shmem();
    test_deposit_charge_analytic();
    test_deposit_charge();
    test_check_boundaries_unit();
    test_check_boundaries_many_blocks();
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
    let amount = DeviceBuffer::from_host(&stream, &[n as u32]).unwrap();
    let module = kernels::load(&ctx).unwrap();
    let cfg    = LaunchConfig::for_num_elems(n as u32);
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        &amount, FACTOR_E as Real, DT_E as Real,
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
    let amount = DeviceBuffer::from_host(&stream, &[n as u32]).unwrap();
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        &amount, FACTOR_E as Real, DT_E as Real,
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
    let amount = DeviceBuffer::from_host(&stream, &[n_test as u32]).unwrap();
    module.move_particles(&stream, cfg,
        &efield_dev, &mut x_dev, &mut vx_dev,
        &amount,
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
    let amount = DeviceBuffer::from_host(&stream, &[N as u32]).unwrap();
    
    // dt = 0 freezes positions across launches so particles never leave [0, L].
    let dt = 0.0 as Real;

    // warm-up (avoids JIT / driver overhead in measurements) TODO - verify if needed
    for _ in 0..10 {
        module.move_particles(&stream, cfg,
            &efield_dev, &mut x_shmem, &mut vx_shmem, &amount, factor, dt).unwrap();
        module.OLD_move_particles(&stream, cfg,
            &efield_dev, &mut x_no_shmem, &mut vx_no_shmem, &amount, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();

    // time shared-memory variant
    let t0 = Instant::now();
    for _ in 0..REPS {
        module.move_particles(&stream, cfg,
            &efield_dev, &mut x_shmem, &mut vx_shmem, &amount, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();
    let t_shmem = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6; // µs per launch

    // time no-shared-memory variant
    let t0 = Instant::now();
    for _ in 0..REPS {
        module.OLD_move_particles(&stream, cfg,
            &efield_dev, &mut x_no_shmem, &mut vx_no_shmem, &amount, factor, dt).unwrap();
    }
    ctx.synchronize().unwrap();
    let t_no_shmem = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6;

    println!(">> bench move_particles ({} particles, {} reps):", N, REPS);
    println!("     shared memory : {:.2} µs/launch", t_shmem);
    println!("     no shared mem : {:.2} µs/launch", t_no_shmem);
    println!("     speedup       : {:.2}×", t_no_shmem / t_shmem);
}

// run zero_density + deposit_charge
fn run_deposit_on_gpu(x_host: &[Real]) -> Vec<Real> {
    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();

    let x_dev = DeviceBuffer::from_host(&stream, x_host).unwrap();
    let density_dev = DeviceBuffer::<Real>::zeroed(&stream, N_G).unwrap();

    let cfg_deposit = LaunchConfig::for_num_elems(x_host.len() as u32);
    let amount = DeviceBuffer::from_host(&stream, &[x_host.len() as u32]).unwrap();

    module.get_density(
        &stream, cfg_deposit,
        &x_dev, &density_dev, &amount,
    ).unwrap();

    density_dev.to_host_vec(&stream).unwrap()
}

fn cpu_get_density(x_host: &[Real]) -> Vec<Real> {
    let mut density = vec![0.0 as Real; N_G];
    let c: Real = (WEIGHT / (ELECTRODE_AREA * DX)) as Real;
    for &xi in x_host {
        let pos = xi * INV_DX as Real;
        let q   = pos as usize;
        let rem = pos - q as Real;
        density[q]     += (1.0 as Real - rem) * c;
        density[q + 1] += rem * c;
    }
    density[0]       *= 2.0 as Real;
    density[N_G - 1] *= 2.0 as Real;
    density
}

fn test_deposit_charge_analytic() {
    let c: Real = (WEIGHT / (ELECTRODE_AREA * DX)) as Real;
    let x_host: Vec<Real> = vec![
        100.5 * DX as Real,
        50.0  * DX as Real,
        0.5   * DX as Real,
        ((N_G - 2) as Real + 0.5) * DX as Real,
    ];

    let gpu = run_deposit_on_gpu(&x_host);
    let expected = cpu_get_density(&x_host);

    let eps: Real = if std::mem::size_of::<Real>() == 4 { 1e-4 as Real * c } else { 1e-10 as Real * c };
    let mut errors = 0;
    for i in 0..N_G {
        if (gpu[i] - expected[i]).abs() > eps {
            eprintln!("deposit_analytic[{}]: GPU={:.6e} expected={:.6e}", i, gpu[i], expected[i]);
            errors += 1;
        }
    }
    if errors > 0 {
        println!("test_deposit_charge_analytic: {} mismatches", errors);
        std::process::exit(1);
    }
}

fn test_deposit_charge() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const N_PART: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(0xC2DAu64);

    let x_host: Vec<Real> = (0..N_PART)
        .map(|_| (rng.random::<f64>() * L) as Real)
        .collect();

    let gpu = run_deposit_on_gpu(&x_host);
    let cpu = cpu_get_density(&x_host);

    let sum_gpu: f64 = gpu.iter().map(|&v| v as f64).sum::<f64>() * DX;
    let sum_cpu: f64 = cpu.iter().map(|&v| v as f64).sum::<f64>() * DX;

    let cons_rel = ((sum_gpu - sum_cpu) / sum_cpu).abs();
    if cons_rel > 1e-10 {
        eprintln!("deposit conservation mismatch: GPU sum*DX = {:.6e}, CPU sum*DX = {:.6e}, rel diff = {:.2e}",
            sum_gpu, sum_cpu, cons_rel);
        std::process::exit(1);
    }

    let max_cell = cpu.iter().fold(0.0 as Real, |a, &b| if b > a { b } else { a });
    let tol_abs: Real = (max_cell * 1e-10) as Real;
    let mut errors = 0;
    let mut max_diff: Real = 0.0;
    for i in 0..N_G {
        let d = (gpu[i] - cpu[i]).abs();
        if d > max_diff { max_diff = d; }
        if d > tol_abs {
            if errors < 5 {
                eprintln!("deposit[{}]: GPU={:.6e} CPU={:.6e} diff={:.2e}", i, gpu[i], cpu[i], d);
            }
            errors += 1;
        }
    }
    if errors > 0 {
        println!("test_deposit_charge: {} cells exceed tol={:.2e}, max_diff={:.2e}",
            errors, tol_abs, max_diff);
        std::process::exit(1);
    }
}

// =========================
// check_boundaries tests
// =========================

// run check_boundaries_compact on the GPU.
// returns: (dst_x, dst_vx, dst_vy, dst_vz, n_alive).
// only the first `n_alive` entries of the dst vectors are valid survivors.
fn run_check_boundaries(
    x: &[Real], vx: &[Real], vy: &[Real], vz: &[Real],
) -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, u32) {
    let n = x.len();

    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();

    let src_x  = DeviceBuffer::from_host(&stream, x).unwrap();
    let src_vx = DeviceBuffer::from_host(&stream, vx).unwrap();
    let src_vy = DeviceBuffer::from_host(&stream, vy).unwrap();
    let src_vz = DeviceBuffer::from_host(&stream, vz).unwrap();

    let mut dst_x  = DeviceBuffer::<Real>::zeroed(&stream, n).unwrap();
    let mut dst_vx = DeviceBuffer::<Real>::zeroed(&stream, n).unwrap();
    let mut dst_vy = DeviceBuffer::<Real>::zeroed(&stream, n).unwrap();
    let mut dst_vz = DeviceBuffer::<Real>::zeroed(&stream, n).unwrap();

    let alive   = DeviceBuffer::from_host(&stream, &[0u32]).unwrap();
    let amount  = DeviceBuffer::from_host(&stream, &[n as u32]).unwrap();

    let cfg = LaunchConfig::for_num_elems(n as u32);
    module.check_boundaries_compact(
        &stream, cfg,
        &src_x, &src_vx, &src_vy, &src_vz,
        &mut dst_x, &mut dst_vx, &mut dst_vy, &mut dst_vz,
        &alive, &amount,
    ).unwrap();

    let n_alive = alive.to_host_vec(&stream).unwrap()[0];

    let dx  = dst_x.to_host_vec(&stream).unwrap();
    let dvx = dst_vx.to_host_vec(&stream).unwrap();
    let dvy = dst_vy.to_host_vec(&stream).unwrap();
    let dvz = dst_vz.to_host_vec(&stream).unwrap();

    (dx, dvx, dvy, dvz, n_alive)
}

fn test_check_boundaries_unit() {
    //                  idx:  0     1   2     3   4   5     6   7   8   9
    // fate:                  pow   ok  gnd   ok  ok  pow   ok  gnd ok  gnd
    let x: Vec<Real> = vec![
        -1.0e-3,                 // 0: x < 0           -> abs_pow
        0.5 * L as Real,         // 1: inside          -> alive
        L as Real + 1.0e-3,      // 2: x > L           -> abs_gnd
        0.0,                     // 3: x == 0 (edge)   -> alive
        0.25 * L as Real,        // 4: inside          -> alive
        -5.0e-4,                 // 5: x < 0           -> abs_pow
        L as Real,               // 6: x == L (edge)   -> alive
        2.0 * L as Real,         // 7: x > L           -> abs_gnd
        0.75 * L as Real,        // 8: inside          -> alive
        L as Real + 5.0e-3,      // 9: x > L           -> abs_gnd
    ];

    // unique velocity signatures so survivor data can be verified exactly.
    let vx: Vec<Real> = (0..x.len()).map(|i| (i as Real + 1.0) * 10.0).collect();
    let vy: Vec<Real> = (0..x.len()).map(|i| (i as Real + 1.0) * 100.0).collect();
    let vz: Vec<Real> = (0..x.len()).map(|i| (i as Real + 1.0) * 1000.0).collect();

    let (dx, dvx, dvy, dvz, n_alive) = run_check_boundaries(&x, &vx, &vy, &vz);

    let mut errors = 0;
    if n_alive != 5 { 
        eprintln!("unit alive: got {} expected 5", n_alive); 
        errors += 1; 
    }

    let mut got: Vec<(Real, Real, Real, Real)> =
        (0..n_alive as usize).map(|i| (dx[i], dvx[i], dvy[i], dvz[i])).collect();
    let mut expected: Vec<(Real, Real, Real, Real)> =
        [1usize, 3, 4, 6, 8].iter().map(|&i| (x[i], vx[i], vy[i], vz[i])).collect();
    got.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if got.len() == expected.len() {
        for (g, e) in got.iter().zip(expected.iter()) {
            if g != e {
                eprintln!("unit survivor mismatch: got {:?} expected {:?}", g, e);
                errors += 1;
            }
        }
    }
    if errors > 0 {
        println!("test_check_boundaries_unit: {} errors", errors);
        std::process::exit(1);
    }
}

fn test_check_boundaries_many_blocks() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const N: usize = 100_000;
    let mut rng = StdRng::seed_from_u64(0xB0DEu64);

    // positions in [-0.1 L, 1.1 L] -> ~1/12 absorbed at each electrode
    let x: Vec<Real> = (0..N)
        .map(|_| ((rng.random::<f64>() * 1.2 - 0.1) * L) as Real)
        .collect();

    // unique per-particle velocity keys
    let vx: Vec<Real> = (0..N).map(|i| i as Real).collect();
    let vy: Vec<Real> = (0..N).map(|i| (i as Real) * 2.0).collect();
    let vz: Vec<Real> = (0..N).map(|i| (i as Real) * 3.0).collect();

    // CPU reference
    let mut ref_alive: Vec<(Real, Real, Real, Real)> = Vec::new();
    for i in 0..N {
        if x[i] >= 0.0 as Real && x[i] <= L as Real {
            ref_alive.push((x[i], vx[i], vy[i], vz[i]));
        }
    }

    let (dx, dvx, dvy, dvz, n_alive) = run_check_boundaries(&x, &vx, &vy, &vz);

    let mut errors = 0;
    if n_alive as usize != ref_alive.len() {
        eprintln!("alive: got {} expected {}", n_alive, ref_alive.len());
        errors += 1;
    }

    let mut got: Vec<(Real, Real, Real, Real)> =
        (0..n_alive as usize).map(|i| (dx[i], dvx[i], dvy[i], dvz[i])).collect();
    got.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if got.len() == ref_alive.len() {
        for (g, e) in got.iter().zip(ref_alive.iter()) {
            if g != e {
                if errors < 5 {
                    eprintln!("survivor mismatch: got {:?} expected {:?}", g, e);
                }
                errors += 1;
            }
        }
    }
    if errors > 0 {
        println!("test_check_boundaries: {} errors", errors);
        std::process::exit(1);
    }
}

// =====================
// Poisson solver tests
// =====================

// Flexible DPS test kernel: supports up to 512 grid points (N=8..400).
const DPS_TEST_BLOCK: u32    = 512;
const DPS_TEST_WARPS: usize  = DPS_TEST_BLOCK as usize / 32;    // = 16
const DPS_TEST_MAX_N: usize  = 512;

/// Generic CPU Double-Prefix-Sum solver for ψ''(x) = s(x) on [0,1]
/// with Dirichlet BCs ψ(0)=psi_left, ψ(N-1)=psi_right.
///
/// Discretization: ψ[i-1] - 2ψ[i] + ψ[i+1] = dx² · s(x_i)
/// for interior nodes i=1..N-2.
///
/// This is the same algorithm as the GPU kernel `solve_poisson_scan_f32`,
/// parametrized by grid size N instead of the fixed N_G=400.
fn solve_poisson_dps_generic(n: usize, source: &[f64], psi_left: f64, psi_right: f64) -> Vec<f64> {
    let dx = 1.0 / (n - 1) as f64;

    let mut f = vec![0.0f64; n];
    for i in 1..=(n - 2) {
        f[i] = dx * dx * source[i];
    }
    f[1]     -= psi_left;
    f[n - 2] -= psi_right;

    let mut g = vec![0.0f64; n];
    for i in 1..n {
        g[i] = (i as f64) * f[i];
    }
    let mut s = vec![0.0f64; n];
    for i in 1..n {
        s[i] = s[i - 1] + g[i];
    }

    let mut h = vec![0.0f64; n];
    for i in 1..(n - 1) {
        h[i] = -s[i] / (i as f64 + 1.0);
    }

    let mut r = vec![0.0f64; n];
    for i in 1..(n - 1) {
        r[i] = h[i] / (i as f64);
    }
    let mut big_r = vec![0.0f64; n];
    for i in 1..n {
        big_r[i] = big_r[i - 1] + r[i];
    }
    let total = big_r[n - 2];

    let mut psi = vec![0.0f64; n];
    psi[0]     = psi_left;
    psi[n - 1] = psi_right;
    for i in 1..(n - 1) {
        psi[i] = (i as f64) * (total - big_r[i] + r[i]);
    }
    psi
}

// Convergence test for Double-Prefix-Sum Poisson solver.
// (https://ammar-hakim.org/sj/je/je11/je11-fem-poisson.html#convergence-of-1d-solver) 
//
// Problem: ψ''(x) = 1 - 2x² on [0,1], ψ(0)=ψ(1)=0
// Exact:   ψ(x) = x²/2 - x⁴/6 - x/3
//
// Expected: second-order convergence (error ∝ dx²), order → 2.0.
// Writes results to `results/dps_cpu_N{n}.csv`.
fn test_dps_convergence() {
    use std::fs;
    use std::io::Write;

    let a: f64 = 2.0;
    let source_fn = |x: f64| -> f64 { 1.0 - a * x * x };
    let exact_fn  = |x: f64| -> f64 { x * x / 2.0 - x.powi(4) / 6.0 - x / 3.0 };

    let grid_sizes: &[usize] = &[8, 16, 32, 64, 100, 200, 400];
    let mut errors: Vec<f64> = Vec::new();
    let mut dxs: Vec<f64> = Vec::new();

    fs::create_dir_all("results").unwrap();

    println!(">> test_dps_convergence: ψ''(x) = 1 - 2x², ψ(0)=ψ(1)=0");
    println!("   {:>4}  {:>12}  {:>12}  {:>8}", "N", "dx", "avg_error", "order");

    for &n in grid_sizes {
        let dx = 1.0 / (n - 1) as f64;
        dxs.push(dx);

        let source: Vec<f64> = (0..n).map(|i| source_fn(i as f64 * dx)).collect();
        let psi_num = solve_poisson_dps_generic(n, &source, 0.0, 0.0);

        let mut file = fs::File::create(format!("results/dps_cpu_N{}.csv", n)).unwrap();
        writeln!(file, "x,psi_num,psi_exact").unwrap();
        for i in 0..n {
            let x = i as f64 * dx;
            writeln!(file, "{:.15e},{:.15e},{:.15e}", x, psi_num[i], exact_fn(x)).unwrap();
        }

        let mut err_sum = 0.0f64;
        let mut count = 0usize;
        for i in 1..(n - 1) {
            let x = i as f64 * dx;
            err_sum += (psi_num[i] - exact_fn(x)).abs();
            count += 1;
        }
        let avg_err = err_sum / count as f64;
        errors.push(avg_err);

        let order = if errors.len() >= 2 {
            (errors[errors.len() - 2] / errors[errors.len() - 1]).ln()
                / (dxs[dxs.len() - 2] / dxs[dxs.len() - 1]).ln()
        } else { 0.0 };

        if errors.len() >= 2 {
            println!("   {:>4}  {:>12.6e}  {:>12.6e}  {:>8.4}", n, dx, avg_err, order);
        } else {
            println!("   {:>4}  {:>12.6e}  {:>12.6e}  {:>8}", n, dx, avg_err, "---");
        }
    }

    let final_order = (errors[errors.len() - 2] / errors[errors.len() - 1]).ln()
        / (dxs[dxs.len() - 2] / dxs[dxs.len() - 1]).ln();

    let mut test_errors = 0;
    for i in 1..errors.len() {
        if errors[i] >= errors[i - 1] { test_errors += 1; }
    }
    if final_order < 1.8 { test_errors += 1; }
    if errors[errors.len() - 1] > 1e-4 { test_errors += 1; }

    if test_errors > 0 {
        println!("test_dps_convergence: FAILED ({} assertions)", test_errors);
        std::process::exit(1);
    }
    println!("   → PASSED (order={:.4}), files written to results/dps_cpu_N*.csv", final_order);
}

// GPU convergence test — writes results to `results/dps_gpu_N{n}.csv`.
fn test_dps_convergence_gpu() {
    use std::fs;
    use std::io::Write;

    let a: f64 = 2.0;
    let source_fn = |x: f64| -> f64 { 1.0 - a * x * x };
    let exact_fn  = |x: f64| -> f64 { x * x / 2.0 - x.powi(4) / 6.0 - x / 3.0 };

    let ctx    = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (DPS_TEST_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let grid_sizes: &[usize] = &[8, 16, 32, 64, 100, 200, 400];
    let mut errors: Vec<f64> = Vec::new();
    let mut dxs: Vec<f64> = Vec::new();

    fs::create_dir_all("results").unwrap();

    println!(">> test_dps_convergence_gpu: ψ''(x) = 1 - 2x², ψ(0)=ψ(1)=0  [GPU kernel f32]");
    println!("   {:>4}  {:>12}  {:>12}  {:>8}", "N", "dx", "avg_error", "order");

    for &n in grid_sizes {
        let dx = 1.0 / (n - 1) as f64;
        dxs.push(dx);

        let source_f32: Vec<f32> = (0..n).map(|i| source_fn(i as f64 * dx) as f32).collect();
        let source_dev = DeviceBuffer::from_host(&stream, &source_f32).unwrap();
        let mut pot_dev = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

        module.solve_poisson_dps_flexible(
            &stream, cfg,
            &source_dev, &mut pot_dev,
            n as u32, 0.0f32, 0.0f32, dx as f32,
        ).unwrap();

        let pot_gpu = pot_dev.to_host_vec(&stream).unwrap();

        let mut file = fs::File::create(format!("results/dps_gpu_N{}.csv", n)).unwrap();
        writeln!(file, "x,psi_num,psi_exact").unwrap();
        for i in 0..n {
            let x = i as f64 * dx;
            writeln!(file, "{:.15e},{:.15e},{:.15e}", x, pot_gpu[i] as f64, exact_fn(x)).unwrap();
        }

        let mut err_sum = 0.0f64;
        for i in 1..(n - 1) {
            let x = i as f64 * dx;
            err_sum += (pot_gpu[i] as f64 - exact_fn(x)).abs();
        }
        let avg_err = err_sum / (n - 2) as f64;
        errors.push(avg_err);

        let order = if errors.len() >= 2 {
            (errors[errors.len() - 2] / errors[errors.len() - 1]).ln()
                / (dxs[dxs.len() - 2] / dxs[dxs.len() - 1]).ln()
        } else { 0.0 };

        if errors.len() >= 2 {
            println!("   {:>4}  {:>12.6e}  {:>12.6e}  {:>8.4}", n, dx, avg_err, order);
        } else {
            println!("   {:>4}  {:>12.6e}  {:>12.6e}  {:>8}", n, dx, avg_err, "---");
        }
    }

    let final_order = (errors[errors.len() - 2] / errors[errors.len() - 1]).ln()
        / (dxs[dxs.len() - 2] / dxs[dxs.len() - 1]).ln();

    let mut test_errors = 0;
    for i in 1..errors.len() {
        if errors[i] >= errors[i - 1] { test_errors += 1; }
    }
    if final_order < 1.8 { test_errors += 1; }
    if errors[errors.len() - 1] > 1e-4 { test_errors += 1; }

    if test_errors > 0 {
        println!("test_dps_convergence_gpu: FAILED ({} assertions)", test_errors);
        std::process::exit(1);
    }
    println!("   → PASSED (order={:.4}), files written to results/dps_gpu_N*.csv", final_order);
}