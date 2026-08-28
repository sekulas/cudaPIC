//-------------------------------------------------------------------//
//       cudaPIC : GPU-parallel 1d3v PIC/MCC simulation           //
//       Based on eduPIC by Z. Donko et al. (2021)                   //
//       Parallelized with cuda-oxide for NVIDIA GPUs                //
//-------------------------------------------------------------------//

#![allow(non_snake_case)]
#![allow(dead_code)]

use cuda_core::{CudaContext, DeviceBuffer, PinnedHostBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, ptx_asm, DisjointSlice, SharedArray, warp, gpu_assert};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicF32, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::cooperative_groups::{block_scan, ops::Sum, this_thread_block};
use rand::RngExt;
use rand_distr::Normal;
use std::{env, fmt};
use std::time::Instant;
use std::io::BufWriter;
use std::fs::{self, File};
use std::io::Write;
use chrono::{DateTime};
use chrono_tz::Europe::Warsaw;
use chrono_tz::Tz;

type Real = f32;

// constants
const PI: f64              = 3.141592653589793;      // mathematical constant Pi
const TWO_PI: f64          = 2.0 * PI;               // two times Pi
const E_CHARGE: f64        = 1.60217662e-19;         // electron charge [C]
const E_MASS: f64          = 9.10938356e-31;         // mass of electron [kg]
const AR_MASS: f64         = 6.63352090e-26;         // mass of argon atom [kg]
const MU_ARAR: f64         = AR_MASS / 2.0;          // reduced mass of two argon atoms [kg]
const K_BOLTZMANN: f64     = 1.38064852e-23;         // boltzmann's constant [J/K]
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
const GAS_DENSITY: f64     = PRESSURE / (K_BOLTZMANN * TEMPERATURE);   // background gas density [m-3]

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

// gpu capacity and allocation constants
const MAX_PARTICLES: usize = 120_000;                       // maximum number of particles per species (pre-allocated on GPU).
const FACTOR_E: f64 = DT_E / E_MASS * (-E_CHARGE);          // leapfrog acceleration factor for electrons
const FACTOR_I: f64 = DT_I / AR_MASS * E_CHARGE;            // leapfrog acceleration factor for ions
const WEIGHT_FACTOR: f64 = WEIGHT / (ELECTRODE_AREA * DX);  // weight factor for density deposition

// kernel launch parameters
const MAX_PARTICLES_U32: u32 = MAX_PARTICLES as u32;                            // particle amount for kernel launch param
const POISSON_SCAN_BLOCK_SIZE: u32   = 416;                                     // size of block for poisson kernel 
const POISSON_SCAN_NUM_WARPS:  usize = POISSON_SCAN_BLOCK_SIZE as usize / 32;   // amount of warps in block for poisson kernel
const COMPACT_BLOCK_SIZE: u32   = 512;                                          // size of block for stream compaction kernel
const COMPACT_NUM_WARPS:  usize = COMPACT_BLOCK_SIZE as usize / 32;             // amount of warps in block for stream compaction kernel

// collisions
const NORMAL_RANGE: Real = 269.90040554976775;                                // thermal velocity of background gas
const F1: Real = (E_MASS / (E_MASS + AR_MASS)) as Real;                       // precomputed factor
const F2: Real = (AR_MASS / (E_MASS + AR_MASS)) as Real;                      // precomputed factor
const HALF_E_MASS_OVER_E_CHARGE: Real = (0.5 * E_MASS / E_CHARGE) as Real;    // precomputed factor
const HALF_MU_ARAR_OVER_E_CHARGE: Real = (0.5 * MU_ARAR / E_CHARGE) as Real;  // precomputed factor
const LOG2_E: Real = 1.4426950408889634;                                      // log2(e)

// poisson solver
const E_CHARGE_F: f32 = E_CHARGE as f32;                         // f32 electron charge
const INV_DX_F: f32 = INV_DX as f32;                             // f32 inverse of spatial grid size
const ALPHA_F: f32 = (-DX * DX / EPSILON0) as f32;               // precomputed factor
const HALF_DX_OVER_EPS_F: f32 = (DX / (2.0 * EPSILON0)) as f32;  // precomputed factor
const HALF_INV_DX_F: Real = (0.5 * INV_DX) as Real;              // precomputed factor             
        
// measurment
const MIN_X_F: Real = MIN_X as Real;        // f32 lower limit of central region
const MAX_X_F: Real = MAX_X as Real;        // f32 upper limit of central region
const DE_EEPF_F: Real = DE_EEPF as Real;    // f32 resolution of EEPF
const CHECKPOINT_CYCLES: usize = 100;       // cycles to retrieve number of particles from gpu

// host side particle data container
struct ParticlesSoA {
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
}

// gpu simulation state data
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

    // particle counters
    n_electrons: DeviceBuffer<u32>,
    n_ions:      DeviceBuffer<u32>,

    // grid
    efield:    DeviceBuffer<Real>,
    pot:       DeviceBuffer<Real>,
    e_density: DeviceBuffer<Real>,
    i_density: DeviceBuffer<Real>,

    // cross sections
    cs: DeviceBuffer<Real>,

    // total cross sections
    sigma_tot_e: DeviceBuffer<Real>,
    sigma_tot_i: DeviceBuffer<Real>,

    // rng state
    rng_e0: DeviceBuffer<u32>,
    rng_e1: DeviceBuffer<u32>,
    rng_e2: DeviceBuffer<u32>,
    rng_e3: DeviceBuffer<u32>,
    rng_i0: DeviceBuffer<u32>,
    rng_i1: DeviceBuffer<u32>,
    rng_i2: DeviceBuffer<u32>,
    rng_i3: DeviceBuffer<u32>,

    // buffers for stream compaction
    tmp_x:  DeviceBuffer<Real>,
    tmp_vx: DeviceBuffer<Real>,
    tmp_vy: DeviceBuffer<Real>,
    tmp_vz: DeviceBuffer<Real>,

    // tmp var to store data about particle counts
    alive_counter: DeviceBuffer<u32>,

    // measurment
    cumul_e_density: DeviceBuffer<f64>, 
    cumul_i_density: DeviceBuffer<f64>,
    eepf_counts:     DeviceBuffer<u32>,
}

impl GpuSimState {
    fn allocate(stream: &cuda_core::CudaStream, measure: bool) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            e_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            e_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,

            i_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            i_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,

            n_electrons: DeviceBuffer::<u32>::zeroed(stream, 1)?,
            n_ions:      DeviceBuffer::<u32>::zeroed(stream, 1)?,

            efield:    DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            pot:       DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            e_density: DeviceBuffer::<Real>::zeroed(stream, N_G)?,
            i_density: DeviceBuffer::<Real>::zeroed(stream, N_G)?,

            cs: DeviceBuffer::<Real>::zeroed(stream, N_CS * CS_RANGES)?,

            sigma_tot_e: DeviceBuffer::<Real>::zeroed(stream, CS_RANGES)?,
            sigma_tot_i: DeviceBuffer::<Real>::zeroed(stream, CS_RANGES)?,

            rng_e0: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e1: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e2: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_e3: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i0: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i1: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i2: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,
            rng_i3: DeviceBuffer::<u32>::zeroed(stream, MAX_PARTICLES)?,

            alive_counter: DeviceBuffer::<u32>::zeroed(stream, 1)?,

            tmp_x:  DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vx: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vy: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,
            tmp_vz: DeviceBuffer::<Real>::zeroed(stream, MAX_PARTICLES)?,

            cumul_e_density: if measure {
                DeviceBuffer::<f64>::zeroed(stream, N_G)?
            } else {
                DeviceBuffer::<f64>::zeroed(stream, 0)?
            },
            cumul_i_density: if measure {
                DeviceBuffer::<f64>::zeroed(stream, N_G)?
            } else {
                DeviceBuffer::<f64>::zeroed(stream, 0)?
            },
            eepf_counts: if measure {
                DeviceBuffer::<u32>::zeroed(stream, N_EEPF)?
            } else {
                DeviceBuffer::<u32>::zeroed(stream, 0)?
            },
        })
    }

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

    fn upload_cross_sections(
        &mut self,
        stream: &cuda_core::CudaStream,
        cs_flat: &[Real],       
        sigma_tot_e: &[Real],
        sigma_tot_i: &[Real],
    ) -> Result<(), cuda_core::DriverError> {
        self.cs.copy_from_host(stream, cs_flat)?;
        self.sigma_tot_e.copy_from_host(stream, sigma_tot_e)?;
        self.sigma_tot_i.copy_from_host(stream, sigma_tot_i)?;
        Ok(())
    }

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

    fn download_measurements(
        &self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<u32>), cuda_core::DriverError> {
        let e_dens = self.cumul_e_density.to_host_vec(stream)?;
        let i_dens = self.cumul_i_density.to_host_vec(stream)?;
        let eepf   = self.eepf_counts.to_host_vec(stream)?;
        Ok((e_dens, i_dens, eepf))
    }
}

#[cuda_module]
mod kernels {
    use super::*;

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

        // shared mem
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

        // global mem flush
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


    #[kernel]
    pub fn solve_poisson_scan_f32(
        e_density:  &[f32],
        i_density:  &[f32],
        mut pot:    DisjointSlice<f32>,
        mut efield: DisjointSlice<f32>,
        pot0:       f32,
    ) {
        static mut SCAN_SMEM: SharedArray<f32, POISSON_SCAN_NUM_WARPS> = SharedArray::UNINIT;
        static mut POT_S:     SharedArray<f32, N_G>            = SharedArray::UNINIT;
        static mut R_TOTAL_S:   SharedArray<f32, 1>              = SharedArray::UNINIT;

        let tid   = thread::threadIdx_x() as usize;
        let block = this_thread_block();

        let rho_i = if tid < N_G {
            E_CHARGE_F * (i_density[tid] - e_density[tid])
        } else {
            0.0f32
        };

        let g_i: f32 = if tid == 0 || tid >= N_G {
            0.0f32
        } else {
            let f_i   = ALPHA_F * rho_i - if tid == 1 { pot0 } else { 0.0f32 };
            (tid as f32) * f_i
        };

        let big_g_i = block_scan::<f32, Sum, _>(&block, g_i, &raw mut SCAN_SMEM);

        thread::sync_threads();

        let r_i: f32 = if tid == 0 || tid >= N_G - 1 {
            0.0f32
        } else {
            let h_i = -big_g_i / (tid as f32 + 1.0f32);
            h_i / (tid as f32)
        };

        let big_r_i = block_scan::<f32, Sum, _>(&block, r_i, &raw mut SCAN_SMEM);

        if tid == N_G - 2 {
            unsafe { R_TOTAL_S[0] = big_r_i; }
        }

        thread::sync_threads();
        let r_total = unsafe { R_TOTAL_S[0] };


        let pot_i: f32 = if tid == 0 {
            pot0
        } else if tid < N_G - 1 {
            (tid as f32) * (r_total - big_r_i + r_i)
        } else {
            0.0f32 
        };

        if tid < N_G {
            unsafe { POT_S[tid] = pot_i; }
        }


        thread::sync_threads();

        if let Some((pot_elem, idx)) = pot.get_mut_indexed() {
            let pk = unsafe { POT_S[tid] };
            *pot_elem = pk;

            let e_i = if tid == 0 {
                let pk1  = unsafe { POT_S[1] };
                (pk - pk1) * INV_DX_F - rho_i * HALF_DX_OVER_EPS_F
            } else if tid == N_G - 1 {
                let pkm1  = unsafe { POT_S[N_G - 2] };
                (pkm1 - pk) * INV_DX_F + rho_i * HALF_DX_OVER_EPS_F
            } else {
                let pkm1 = unsafe { POT_S[tid - 1] };
                let pkp1 = unsafe { POT_S[tid + 1] };
                (pkm1 - pkp1) * HALF_INV_DX_F
            };

            if let Some(efield_elem) = efield.get_mut(idx) {
                *efield_elem = e_i;
            }
        }
    }


    #[kernel]
    pub fn solve_poisson_dps_flexible(
        source:    &[f32],              
        mut pot:   DisjointSlice<f32>,  
        n:         u32,
        psi_left:  f32,                 
        psi_right: f32,                 
        dx:        f32,
    ) {
        static mut SCAN_SMEM: SharedArray<f32, POISSON_SCAN_NUM_WARPS> = SharedArray::UNINIT;
        static mut R_TOTAL_S: SharedArray<f32, 1> = SharedArray::UNINIT;

        let tid   = thread::threadIdx_x() as usize;
        let block = this_thread_block();
        let nn    = n as usize;

        let g_i: f32 = if tid == 0 || tid >= nn {
            0.0f32
        } else {
            let mut f_i = dx * dx * source[tid];
            if tid == 1      { f_i -= psi_left; }
            if tid == nn - 2 { f_i -= psi_right; }
            (tid as f32) * f_i
        };

        let big_g_i = block_scan::<f32, Sum, _>(&block, g_i, &raw mut SCAN_SMEM);

        thread::sync_threads();

        let r_i: f32 = if tid == 0 || tid >= nn - 1 {
            0.0f32
        } else {
            let h_i = -big_g_i / (tid as f32 + 1.0f32);
            h_i / (tid as f32)
        };

        let big_r_i = block_scan::<f32, Sum, _>(&block, r_i, &raw mut SCAN_SMEM);

        if tid == nn - 2 {
            unsafe { R_TOTAL_S[0] = big_r_i; }
        }

        thread::sync_threads();
        let r_total = unsafe { R_TOTAL_S[0] };

        let pot_i: f32 = if tid == 0 {
            psi_left
        } else if tid < nn - 1 {
            (tid as f32) * (r_total - big_r_i + r_i)
        } else if tid == nn - 1 {
            psi_right
        } else {
            0.0f32
        };

        thread::sync_threads();

        if let Some((pot_elem, _idx)) = pot.get_mut_indexed() {
            if tid < nn {
                *pot_elem = pot_i;
            }
        }
    }

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
        alive_counter: &[u32],
        n_active:      &[u32],
        factor:   Real,
        dt:       Real,
    ) {        
        static mut WARP_SUMS: SharedArray<u32, COMPACT_NUM_WARPS> = SharedArray::UNINIT;
        static mut BASE_S:    SharedArray<u32, 1>                 = SharedArray::UNINIT;

        let i     = thread::index_1d().get();
        let lane  = warp::lane_id();
        let wid   = warp::warp_id() as usize;
        let mut flag: bool = false;
        let (mut new_x, mut new_vx, mut vyi, mut vzi): (Real, Real, Real, Real) =
            (0.0, 0.0, 0.0, 0.0);

        // move
        if i < n_active[0] as usize {
            let xi  = src_x[i];
            let vxi = src_vx[i];

            let pos = xi * INV_DX as Real;
            let p   = pos as usize;
            let c2  = pos - p as Real;
            let e_x = (1.0 as Real - c2) * efield[p] + c2 * efield[p + 1];

            new_vx = vxi + factor * e_x;
            new_x  = xi  + new_vx * dt;

            if new_x >= 0.0 as Real && new_x <= L as Real {
                flag = true;
                vyi = src_vy[i];
                vzi = src_vz[i];
            }
        }

        // compact
        let mask        = warp::ballot(flag);
        let lane_offset = (mask & warp::lanemask_lt()).count_ones();
        let warp_total  = mask.count_ones();

        if lane == 0 {
            unsafe { WARP_SUMS[wid] = warp_total; }
        }
        thread::sync_threads();
        if wid == 0 {
            let v: u32 = if (lane as usize) < COMPACT_NUM_WARPS 
                            { unsafe { WARP_SUMS[lane as usize] } } 
                         else 
                            { 0 };

            let mut val = v;
            let mut offset = 1u32;
            while offset < 32 {
                let n = warp::shuffle_up(val, offset);
                if lane >= offset {
                    val += n;
                }
                offset *= 2;
            }
            let excl = val - v;

            if (lane as usize) < COMPACT_NUM_WARPS {
                unsafe { WARP_SUMS[lane as usize] = excl; } // exclusive
            }

            gpu_assert!(COMPACT_NUM_WARPS <= 32, "COMPACT_NUM_WARPS must be ≤ 32 for single-warp prefix scan");
            if lane as usize == COMPACT_NUM_WARPS - 1 {
                let total_block = val; // inclusive
                let base = 
                    if total_block > 0 {
                        let g = unsafe { DeviceAtomicU32::from_ptr(alive_counter.as_ptr() as *mut u32) };
                        g.fetch_add(total_block, AtomicOrdering::Relaxed)
                    } else { 0 };
                unsafe { BASE_S[0] = base; }
            }
        }
        thread::sync_threads();

        if flag {
            let warp_base = unsafe { WARP_SUMS[wid] };
            let base      = unsafe { BASE_S[0] };
            let slot = base as usize + warp_base as usize + lane_offset as usize;
            unsafe {
                *dst_x.get_unchecked_mut(slot)  = new_x;
                *dst_vx.get_unchecked_mut(slot) = new_vx;
                *dst_vy.get_unchecked_mut(slot) = vyi;
                *dst_vz.get_unchecked_mut(slot) = vzi;
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

    #[inline(always)]
    fn ptx_exp(x: Real) -> Real {
        ptx_ex2(x * LOG2_E)
    }

    #[inline(always)]
    fn ptx_tan(x: Real) -> Real {
        let s = ptx_sin(x);
        let c = ptx_cos(x);
        s * ptx_rcp(c)
    }

    #[inline(always)]
    fn ptx_atan_poly(z: Real) -> Real {
        let z2 = z * z;
        ((-0.0464964749 * z2 + 0.15931422) * z2 - 0.327622764) * z2 * z + z
    }

    #[inline(always)]
    fn ptx_atan(x: Real) -> Real {
        let ax = ptx_abs(x);
        let swap = ax > 1.0;
        let z = if swap { ptx_rcp(ax) } else { ax };

        let mut r = ptx_atan_poly(z);
        if swap {
            r = (0.5 * PI as Real) - r;
        }
        if x < 0.0 { -r } else { r }
    }

    // https://math.stackexchange.com/a/1105038
    #[inline(always)]
    fn ptx_atan2(y: Real, x: Real) -> Real {
        let ax = ptx_abs(x);
        let ay = ptx_abs(y);

        let a = if ay > ax { ax * ptx_rcp(ay) } else { ay * ptx_rcp(ax) };
        let mut r = ptx_atan_poly(a);

        if ay > ax { r = (0.5 * PI as Real) - r; }
        if x < 0.0 { r = PI as Real - r; }
        if y < 0.0 { -r } else { r }
    }

    #[inline(always)]
    fn ptx_acos(x: Real) -> Real {
        let s = ptx_sqrt(ptx_abs(1.0 - x * x));
        ptx_atan2(s, x)
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

    fn xoshiro128p_next(mut s0: u32, mut s1: u32, mut s2: u32, mut s3: u32) -> (u32, u32, u32, u32, u32) {
        let result = s0 + s3;
        let t = s1 << 9;

        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        
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
        sigma: Real,
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

        let v2 = unsafe { *vx.get_unchecked_mut(i) * *vx.get_unchecked_mut(i) 
            + *vy.get_unchecked_mut(i) * *vy.get_unchecked_mut(i) 
            + *vz.get_unchecked_mut(i) * *vz.get_unchecked_mut(i) 
        };
        let energy: Real = HALF_E_MASS_OVER_E_CHARGE * v2;
        let c1 = (energy / (DE_CS as Real) + 0.5) as usize;
        let c2 = CS_RANGES - 1;

        let energy_index = c1.min(c2);
        let velocity: Real = ptx_sqrt(v2);
        let nu: Real = total_cs_e[energy_index] * velocity;

        let rand_val = rng_next_f32(i, &mut rng0, &mut rng1, &mut rng2, &mut rng3);
        
        let p_coll: Real = 1.0 - ptx_exp(-nu * DT_E as Real);
        if rand_val < p_coll {
            collision_e(cs, &mut x, &mut vx, &mut vy, &mut vz, i, energy_index, &mut rng0, &mut rng1, &mut rng2, &mut rng3,
                       &mut i_x, &mut i_vx, &mut i_vy, &mut i_vz, alive_e, alive_i);
        }
        
    }

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
            if gz > 0.0 as Real { phi = 0.5 * PI as Real; }
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
        if rnd < t0 / t2 {                                                      // elastic scattering
            chi = ptx_acos(1.0 - 2.0 * rng_next_f32(i, rng0, rng1, rng2, rng3));
            eta = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);
        } else if rnd < t1 / t2 {                                               // excitation
            let mut energy = 0.5 * E_MASS as Real * g * g;
            energy = ptx_abs(energy - E_EXC_TH as Real * E_CHARGE as Real);
            g = ptx_sqrt(2.0 as Real * energy / E_MASS as Real);
            chi = ptx_acos(1.0 - 2.0 * rng_next_f32(i, rng0, rng1, rng2, rng3));
            eta = TWO_PI as Real * rng_next_f32(i, rng0, rng1, rng2, rng3);
        } else {                                                                // ionization
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
            let (vxa, vya, vza) = rng_next_three_normal(i, NORMAL_RANGE, rng0, rng1, rng2, rng3);
            unsafe {
                *i_x.get_unchecked_mut(ion_idx as usize) = *x.get_unchecked_mut(i);
                *i_vx.get_unchecked_mut(ion_idx as usize) = vxa;
                *i_vy.get_unchecked_mut(ion_idx as usize) = vya;
                *i_vz.get_unchecked_mut(ion_idx as usize) = vza;
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
        let g = ptx_sqrt(g2);
        let nu: Real = total_cs_i[energy_index] * g;

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

    #[kernel]
    pub fn accumulate_density(
        density: &[Real],
        mut cumul: DisjointSlice<f64>,
    ) {
        if let Some((c, idx)) = cumul.get_mut_indexed() {
            *c += density[idx.get()] as f64;
        }
    }

    #[kernel]
    pub fn accumulate_eepf(
        x:        &[Real],
        vx:       &[Real],
        vy:       &[Real],
        vz:       &[Real],
        efield:   &[Real],
        active_e: &[u32],
        mut eepf: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= active_e[0] as usize {
            return;
        }

        let xi = x[i];
        if xi <= MIN_X_F || xi >= MAX_X_F {
            return;
        }

        let pos = xi * INV_DX as Real;
        let p   = pos as usize;
        let c2  = pos - p as Real;
        let e_x = (1.0 as Real - c2) * efield[p] + c2 * efield[p + 1];

        let mean_vx = vx[i] - 0.5 * e_x * (DT_E * E_CHARGE / E_MASS) as Real;
        let vyi = vy[i];
        let vzi = vz[i];

        let v_sqr: Real = mean_vx * mean_vx + vyi * vyi + vzi * vzi;
        let energy: Real = HALF_E_MASS_OVER_E_CHARGE * v_sqr;

        let bin = (energy / DE_EEPF_F) as usize;
        if bin < N_EEPF {
            let elem = unsafe { eepf.get_unchecked_mut(bin) };
            let a = unsafe { DeviceAtomicU32::from_ptr(elem as *mut u32) };
            a.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub fn xoshiro128p_test_wrapper(
        s0: u32,
        s1: u32,
        s2: u32,
        s3: u32,
        mut dest: DisjointSlice<u32>,
    ) {
        let (result, ns0, ns1, ns2, ns3) = xoshiro128p_next(s0, s1, s2, s3);
        unsafe {
            *dest.get_unchecked_mut(0) = result;
            *dest.get_unchecked_mut(1) = ns0;
            *dest.get_unchecked_mut(2) = ns1;
            *dest.get_unchecked_mut(3) = ns2;
            *dest.get_unchecked_mut(4) = ns3;
        }
    }
}

// host side init helpers
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
        1.15e-18 * e_lab.powf(-0.1) * (1.0 + 0.015 / e_lab).powf(0.6)
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
        let sum_e = cs_flat[E_ELA * CS_RANGES + i] as f64
                  + cs_flat[E_EXC * CS_RANGES + i] as f64
                  + cs_flat[E_ION * CS_RANGES + i] as f64;

        let sum_i = cs_flat[I_ISO * CS_RANGES + i] as f64
                  + cs_flat[I_BACK * CS_RANGES + i] as f64;

        sigma_tot_e[i] = (sum_e * GAS_DENSITY) as Real;
        sigma_tot_i[i] = (sum_i * GAS_DENSITY) as Real;
    }

    (cs_flat, sigma_tot_e, sigma_tot_i)
}

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

// host side data saving helpers 
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

fn save_particle_data(particles: &ParticlesSoA, amount: usize, step: usize, species: ParticleSpecies, tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();

    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("unable to create directory");
    let filename = format!("{}/{:04}_{}_{}.csv", dir_path, step, time_stamp, species);

    let mut file = File::create(&filename).expect("unable to create file");
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "x,vx,vy,vz").expect("unable to write header");
    
    for i in 0..amount {
        writeln!(
            writer,
            "{},{},{},{}",
            particles.x[i], particles.vx[i], particles.vy[i], particles.vz[i]
        )
        .expect("unable to write particle data");
    }
}

fn save_particle_growth_data(n_e: Vec<u32>, n_i: Vec<u32>, tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();

    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("unable to create directory");
    let filename = format!("{}/particle_growth_{}.csv", dir_path, time_stamp);

    let mut file = File::create(&filename).expect("unable to create file");
    writeln!(file, "step,n_e,n_i").expect("unable to write header");

    for (step, (&n_e_val, &n_i_val)) in n_e.iter().zip(n_i.iter()).enumerate() {
        writeln!(file, "{},{},{}", step*CHECKPOINT_CYCLES, n_e_val, n_i_val).expect("unable to write particle growth data");
    }
}

fn save_density_avg(cumul_e: &[f64], cumul_i: &[f64], n_steps_e: f64, n_steps_i: f64, tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();
    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("unable to create directory");
    let filename = format!("{}/density_avg_{}.csv", dir_path, time_stamp);

    let mut file = File::create(&filename).expect("unable to create file");
    writeln!(file, "x,n_e,n_i").expect("header");
    for k in 0..N_G {
        let x = k as f64 * DX as f64;
        writeln!(file, "{},{},{}", x, cumul_e[k] / n_steps_e as f64, cumul_i[k] / n_steps_i as f64)
            .expect("row");
    }
}

fn save_eepf(eepf_raw: &[u32], tsmp: DateTime<Tz>) {
    let time_stamp = tsmp.format("%Y-%m-%d_%H-%M-%S").to_string();
    let dir_path = format!("results/{}", time_stamp);
    fs::create_dir_all(&dir_path).expect("unable to create directory");
    let filename = format!("{}/eepf_{}.csv", dir_path, time_stamp);

    let h: f64 = eepf_raw.iter().map(|&c| c as f64).sum::<f64>() * DE_EEPF;

    let mut file = File::create(&filename).expect("unable to create file");
    writeln!(file, "energy_eV,eepf").expect("header");
    for (i, &count) in eepf_raw.iter().enumerate() {
        let e = (0.5 + i as f64) * DE_EEPF;
        let val = count as f64 / h / e.sqrt();
        writeln!(file, "{},{}", e, val).expect("row");
    }
}

fn main() {
    // perform_tests();

    println!(">> cudaPIC: starting...");
    println!(">> cudaPIC: cuda-oxide parallel PIC/MCC simulation");
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: cudapic <num_cycles> [--measure <measurement_cycles>]");
        std::process::exit(1);
    }
    let num_cycles: usize = args[1].parse().expect("invalid cycle count");
    let measure: bool = args.get(2).map_or(false, |arg| arg == "--measure");
    let measurement_cycles: usize = args.get(3)
        .map(|s| s.parse().expect("invalid measurement_cycles"))
        .unwrap_or(1);
    let measurement_start_cycle = num_cycles.saturating_sub(measurement_cycles);

    if measure {
        println!(">> cudaPIC: num_cycles = {}, measure = {}, measurement_start_cycle = {}",
        num_cycles, measure, measurement_start_cycle);
    }
    else {
        println!(">> cudaPIC: num_cycles = {}, measure = {}", num_cycles, measure);
    }

    let start_init = Instant::now();

    // cuda context init
    let ctx = CudaContext::new(0).expect("failed to create CUDA context (no GPU?)");
    let stream = ctx.default_stream();
    println!(">> cudaPIC: CUDA context initialized");

    // cpu cs precomputation
    let (cs_flat, sigma_tot_e, sigma_tot_i) = init_cross_sections();
    println!(">> cudaPIC: cross-sections computed ({} entries per process)", CS_RANGES);

    // cpu particle init
    let electrons_host = init_particles(N_INIT);
    let ions_host      = init_particles(N_INIT);

    // gpu state allocation
    let mut gpu = GpuSimState::allocate(&stream, measure)
        .expect("failed to allocate GPU memory");

    // gpu data upload
    gpu.upload_electrons(&stream, &electrons_host, N_INIT as u32)
        .expect("failed to upload electrons");
    gpu.upload_ions(&stream, &ions_host, N_INIT as u32)
        .expect("failed to upload ions");
    gpu.upload_cross_sections(&stream, &cs_flat, &sigma_tot_e, &sigma_tot_i)
        .expect("failed to upload cross-sections");

    let e_seeds = xoshiro128_seed_streams([0x1234_5678, 0x1111_2222, 0x2222_3333, 0x3333_4444], MAX_PARTICLES);
    let i_seeds = xoshiro128_seed_streams([0x4444_5555, 0x5555_6666, 0x6666_7777, 0x7777_8888], MAX_PARTICLES);
    gpu.upload_rng_state(&stream, &e_seeds, &i_seeds)
        .expect("failed to upload RNG state");

    println!(">> cudaPIC: data uploaded to GPU");

    // launch configs definition
    let cfg = LaunchConfig::for_num_elems(MAX_PARTICLES_U32);
    let poisson_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (POISSON_SCAN_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mac_and_density_cfg = LaunchConfig {
        grid_dim: ((MAX_PARTICLES_U32.div_ceil(COMPACT_BLOCK_SIZE)), 1, 1),
        block_dim: (COMPACT_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let acc_density_cfg = poisson_cfg;
    let eepf_cfg    = cfg;         

    // pinned mem for particle count retrieval
    let mut h_counter_e = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();
    let mut h_counter_i = PinnedHostBuffer::<u32>::zeroed(&ctx, 1).unwrap();

    // simulation loop
    println!(">> cudaPIC: running {} cycles x{} steps...", num_cycles, N_T);
    let module = kernels::load(&ctx).expect("Failed to load CUDA module");

    let mut n_e: u32 = N_INIT as u32;
    let mut n_i: u32 = N_INIT as u32;
    let mut n_e_history: Vec<u32> = Vec::with_capacity(num_cycles / CHECKPOINT_CYCLES + 1);
    let mut n_i_history: Vec<u32> = Vec::with_capacity(num_cycles / CHECKPOINT_CYCLES + 1);
    n_e_history.push(n_e);
    n_i_history.push(n_i);

    let cfg_e = cfg;
    let cfg_i = cfg;

    if measure {
        gpu.cumul_e_density.zero_async(&stream).expect("zero cumul_e_density");
        gpu.cumul_i_density.zero_async(&stream).expect("zero cumul_i_density");
        gpu.eepf_counts.zero_async(&stream).expect("zero eepf_counts");
    }

    println!("initialization time: {:.3} s", start_init.elapsed().as_secs_f64());

    let start = Instant::now();
    
    for cycle in 0..num_cycles {
        let in_measurement_window = measure && cycle >= measurement_start_cycle;

        for t in 0..N_T {
            gpu.e_density.zero_async(&stream).expect("Failed to zero e_density");
            module.get_density(&stream, mac_and_density_cfg,
                &gpu.e_x, &gpu.e_density, &gpu.n_electrons,
            ).expect("get_density (electrons) failed");

            if t % N_SUB == 0 {
                gpu.i_density.zero_async(&stream).expect("Failed to zero i_density");
                module.get_density(&stream, mac_and_density_cfg,
                    &gpu.i_x, &gpu.i_density, &gpu.n_ions,
                ).expect("get_density (ions) failed");
            }

            let pot0 = (VOLTAGE * ((t as f64 / N_T as f64) * TWO_PI ).cos() as f64) as f32;
            module.solve_poisson_scan_f32(&stream, poisson_cfg,
                &gpu.e_density, &gpu.i_density, &mut gpu.pot, &mut gpu.efield, pot0,
            ).expect("solve_poisson failed");

            if in_measurement_window {
                module.accumulate_density(&stream, acc_density_cfg,
                    &gpu.e_density, &mut gpu.cumul_e_density,
                ).expect("accumulate_density (e) failed");

                if t % N_SUB == 0 {
                    module.accumulate_density(&stream, acc_density_cfg,
                        &gpu.i_density, &mut gpu.cumul_i_density,
                    ).expect("accumulate_density (i) failed");
                }
            }

            gpu.alive_counter.zero_async(&stream).expect("failed to zero alive_counter");
            module.move_and_compact(&stream, mac_and_density_cfg, &gpu.efield,
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
                module.move_and_compact(&stream, mac_and_density_cfg, &gpu.efield,
                    &gpu.i_x, &gpu.i_vx, &gpu.i_vy, &gpu.i_vz,
                    &mut gpu.tmp_x, &mut gpu.tmp_vx, &mut gpu.tmp_vy, &mut gpu.tmp_vz,
                    &gpu.alive_counter, &gpu.n_ions,  FACTOR_I as Real, DT_I as Real
                ).expect("move_and_compact (ions) failed");
                std::mem::swap(&mut gpu.i_x,  &mut gpu.tmp_x);
                std::mem::swap(&mut gpu.i_vx, &mut gpu.tmp_vx);
                std::mem::swap(&mut gpu.i_vy, &mut gpu.tmp_vy);
                std::mem::swap(&mut gpu.i_vz, &mut gpu.tmp_vz);
                gpu.n_ions.copy_from_device_async(&gpu.alive_counter, &stream).expect("failed to copy alive_counter to n_ions");
                gpu.alive_counter.copy_from_device_async(&gpu.n_electrons, &stream).expect("failed to copy n_electrons to alive_counter");
            }

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

            if in_measurement_window {
                module.accumulate_eepf(&stream, eepf_cfg,
                    &gpu.e_x, &gpu.e_vx, &gpu.e_vy, &gpu.e_vz,
                    &gpu.efield, &gpu.n_electrons, &mut gpu.eepf_counts,
                ).expect("accumulate_eepf failed");
            }
        }

        if (cycle + 1) % CHECKPOINT_CYCLES == 0 {
            unsafe { gpu.n_electrons.copy_to_pinned_host_async(&stream, &mut h_counter_e) }
                .expect("copy_to_pinned_host_async n_electrons checkpoint failed");
            unsafe { gpu.n_ions.copy_to_pinned_host_async(&stream, &mut h_counter_i) }
                .expect("copy_to_pinned_host_async n_ions checkpoint failed");
            
            stream.synchronize().unwrap();

            println!("   checkpoint at cycle {}: n_e={}, n_i={}, time={:.3}s", cycle + 1, 
                h_counter_e[0], h_counter_i[0], 
                start.elapsed().as_secs_f64());
            n_e_history.push(h_counter_e[0]);
            n_i_history.push(h_counter_i[0]);
        }
    }

    // synchronization and result gathering
    ctx.synchronize().expect("CUDA synchronization failed");

    let (_electrons_result, n_e_final) = gpu.download_electrons(&stream)
        .expect("failed to download electrons");
    let (_ions_result, n_i_final) = gpu.download_ions(&stream)
        .expect("failed to download ions");

    let elapsed = start.elapsed().as_secs_f64();
    println!(">> cudaPIC: simulation complete in {:.3} s", elapsed);
    println!(">> cudaPIC: final particles: {} electrons, {} ions", n_e_final, n_i_final);

    let tsmp = chrono::Utc::now().with_timezone(&Warsaw);
    save_particle_data(&_electrons_result, n_e_final as usize, num_cycles, ParticleSpecies::Electrons, tsmp);
    save_particle_data(&_ions_result, n_i_final as usize, num_cycles, ParticleSpecies::Ions, tsmp);
    save_particle_growth_data(n_e_history, n_i_history, tsmp);

    if measure {
        let (cumul_e, cumul_i, eepf_raw) = gpu.download_measurements(&stream)
            .expect("failed to download measurements");

        let n_steps_e = (measurement_cycles as f64 * N_T as f64) as f64;
        let n_steps_i = (measurement_cycles as f64 * (N_T as f64 / N_SUB as f64)) as f64;

        save_density_avg(&cumul_e, &cumul_i, n_steps_e, n_steps_i, tsmp);
        save_eepf(&eepf_raw, tsmp);

        println!(">> cudaPIC: measurement data saved (density_avg, eepf, info) for {} cycles", measurement_cycles);
    }
}

// tests
fn perform_tests() {
    test_gpu_dps_convergence_hakim();
    test_gpu_xoshiro();
    std::process::exit(0);
}

fn hakim_source_fn(x: f64) -> f64 {
    1.0 - 2.0 * x * x
}

fn hakim_exact_fn(x: f64) -> f64 {
    x * x / 2.0 - x.powi(4) / 6.0 - x / 3.0
}

fn test_gpu_dps_convergence_hakim() {
    println!("\n>> TEST: DPS Convergence GPU Hakim (JE11)");
    println!("   source: 1 - 2x^2,  psi(0)=0, psi(1)=0");
    println!("   {:>6}  {:>12}  {:>14}  {:>10}", "N_elem", "dx", "avg_error", "order");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (POISSON_SCAN_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_elements = &[8usize, 16, 32, 64];
    let mut dxs: Vec<f64> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    fs::create_dir_all("results").unwrap();

    for &n_elem in n_elements {
        let n = n_elem + 1;
        let dx = 1.0 / n_elem as f64;
        dxs.push(dx);

        let source_f32: Vec<f32> = (0..n).map(|i| hakim_source_fn(i as f64 * dx) as f32).collect();
        let source_dev = DeviceBuffer::from_host(&stream, &source_f32).unwrap();
        let mut pot_dev = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

        module
            .solve_poisson_dps_flexible(&stream, cfg, &source_dev, &mut pot_dev, n as u32, 0.0f32, 0.0f32, dx as f32)
            .unwrap();

        let pot_gpu = pot_dev.to_host_vec(&stream).unwrap();

        if n_elem == 16 {
            let mut f = fs::File::create("results/hakim_n16_profile.csv").unwrap();
            writeln!(f, "x,psi_numerical,psi_exact").unwrap();
            for i in 0..n {
                let x = i as f64 * dx;
                writeln!(f, "{:.10e},{:.10e},{:.10e}", x, pot_gpu[i] as f64, hakim_exact_fn(x)).unwrap();
            }
        }

        let mut err_sum = 0.0f64;
        for i in 1..(n - 1) {
            let x = i as f64 * dx;
            err_sum += (pot_gpu[i] as f64 - hakim_exact_fn(x)).abs();
        }
        let avg_err = err_sum / (n - 2) as f64;
        errors.push(avg_err);

        let order_str = if errors.len() >= 2 {
            let order = (errors[errors.len() - 2] / errors[errors.len() - 1]).ln()
                / (dxs[dxs.len() - 2] / dxs[dxs.len() - 1]).ln();
            format!("{:.4}", order)
        } else {
            "---".to_string()
        };

        println!("   {:>6}  {:>12.5e}  {:>14.6e}  {:>10}", n_elem, dx, avg_err, order_str);
    }
}

use std::path::Path;

fn load_c_xoshiro_result(path: &Path) -> Vec<u32> {
    let text = fs::read_to_string(path).unwrap();
    let mut values = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: u32 = line
            .parse()
            .map_err(|e| format!("10000-c.txt:{}: invalid u32 '{}': {}", line_no + 1, line, e))
            .unwrap();
        values.push(v);
    }
    values
}

fn test_gpu_xoshiro() {
    println!("\n>> TEST: xoshiro to ref");
    let c_res = load_c_xoshiro_result(Path::new("xorshiro128c/10000-c.txt"));
    let n = c_res.len();

    let (mut s0, mut s1, mut s2, mut s3) = (1, 1, 1, 1);

    let ctx = CudaContext::new(0).expect("failed to create cuda context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("failed to load cuda module");
    let mut dest = DeviceBuffer::<u32>::zeroed(&stream, 5).expect("failed to create device buffer");

    // one block of one thread
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut results = vec![0u32; n];

    for i in 0..n {
        module.xoshiro128p_test_wrapper(&stream, cfg, s0, s1, s2, s3, &mut dest).expect("failed to launch cuda kernel");
        let dest_host = dest.to_host_vec(&stream).expect("failed to copy device buffer to host");
        results[i] = dest_host[0];
        s0 = dest_host[1];
        s1 = dest_host[2];
        s2 = dest_host[3];
        s3 = dest_host[4];
    }

    let mut first_mismatch: Option<usize> = None;
    for i in 0..n {
        if results[i] != c_res[i] {
            first_mismatch = Some(i);
            break;
        }
    }

    match first_mismatch {
        None => {
            println!(
                "   gpu xoshiro matches C reference for all {} values",
                n
            );

            let dir_path = format!("results/xoshiro128p_test");
            fs::create_dir_all(&dir_path).expect("unable to create directory");
            let filename = format!("{}/gpu_to_c_res.csv", dir_path);

            let mut file = File::create(&filename).expect("unable to create file");
            writeln!(file, "gpu,c").expect("unable to write header");

            for (step, (&gpu_val, &c_val)) in results.iter().zip(c_res.iter()).enumerate() {
                writeln!(file, "{},{}", gpu_val, c_val).expect("unable to write row");
            }
        }
        Some(i) => {
            println!("gpu xoshiro diverged from reference at index {}", i);
        }
    }
}