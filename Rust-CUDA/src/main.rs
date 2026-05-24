// switching off some compiler warnings
#![allow(non_snake_case)]
#![allow(dead_code)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

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