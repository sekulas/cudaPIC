#!/bin/bash 
set -euo pipefail

THREADS="${1:-64}"
CYCLES="${2:-2000}"

PROJECT_DIR="$HOME/parallelEduPIC_v1.3"
INPUT_DIR="$PROJECT_DIR/cycle_0"

# 1. Generowanie nazwy folderu (np. run_20260319_231530_t64)
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RUN_DIR="$PROJECT_DIR/results/run_${TIMESTAMP}_t${THREADS}"

# 2. Utworzenie folderu i skopiowanie danych startowych
mkdir -p "$RUN_DIR"
if [ -d "$INPUT_DIR" ]; then
    cp -r "$INPUT_DIR/"* "$RUN_DIR/"
else
    echo "Błąd: Katalog $INPUT_DIR nie istnieje!"
    exit 1
fi

echo "========================================================"
echo " Przygotowano środowisko robocze:"
echo " Katalog : $RUN_DIR"
echo " Wątki   : $THREADS"
echo " Cykle   : $CYCLES"
echo "========================================================"

# 3. Dynamiczne wygenerowanie i wysłanie zadania SLURM
# Używamy konstrukcji '<<EOF', aby przekazać treść prosto do komendy sbatch
sbatch <<EOF
#!/bin/bash -l
#SBATCH -J pic_t${THREADS}
#SBATCH -p plgrid-lem-cpu
#SBATCH -A plgwutpleecs2026
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c ${THREADS}
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=${THREADS}
#SBATCH --mem-bind=local
#SBATCH --output="${RUN_DIR}/slurm_%j.out"
#SBATCH --error="${RUN_DIR}/slurm_%j.err"

module load Rust/1.88.0-GCCcore-14.3.0

# Przejście do dedykowanego folderu dla tego uruchomienia
cd "${RUN_DIR}"

START_TIME=\$(date +"%Y-%m-%d %H:%M:%S")

echo "========================================================"
echo " Start time : \$START_TIME"
echo " Threads    : ${THREADS}"
echo " Cycles     : ${CYCLES}"
echo "========================================================"

# --- Zbieranie informacji o węźle ---
NODE_INFO_FILE="${RUN_DIR}/node_info.txt"
{
    echo "========================================================"
    echo " NODE INFO — \$(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    echo ""
    echo "--- Hostname / kernel ---"
    hostname -f
    uname -a

    echo ""
    echo "--- SLURM environment ---"
    echo "SLURM_JOB_ID         = \${SLURM_JOB_ID:-N/A}"
    echo "SLURM_NODELIST       = \${SLURM_NODELIST:-N/A}"
    echo "SLURM_CPUS_PER_TASK  = \${SLURM_CPUS_PER_TASK:-N/A}"
    echo "SLURM_CPU_BIND       = \${SLURM_CPU_BIND:-N/A}"
    echo "SLURM_CPU_BIND_LIST  = \${SLURM_CPU_BIND_LIST:-N/A}"
    echo "SLURM_MEM_PER_NODE   = \${SLURM_MEM_PER_NODE:-N/A}"
    echo "SLURM_JOB_PARTITION  = \${SLURM_JOB_PARTITION:-N/A}"
    echo "OMP_NUM_THREADS      = \${OMP_NUM_THREADS:-N/A}"

    echo ""
    echo "--- CPU topology (lscpu) ---"
    lscpu

    echo ""
    echo "--- NUMA hardware ---"
    numactl --hardware 2>/dev/null || echo "(numactl not available)"

    echo ""
    echo "--- NUMA binding (current process) ---"
    numactl --show 2>/dev/null || echo "(numactl not available)"

    echo ""
    echo "--- CPU affinity (this shell) ---"
    taskset -cp \$\$ 2>/dev/null || echo "(taskset not available)"

    echo ""
    echo "--- CPU frequency governor ---"
    for cpu_dir in /sys/devices/system/cpu/cpu{0,1,2,3}; do
        gov="\${cpu_dir}/cpufreq/scaling_governor"
        freq="\${cpu_dir}/cpufreq/scaling_cur_freq"
        [ -f "\$gov" ]  && echo "\$(basename \$cpu_dir) governor : \$(cat \$gov)"
        [ -f "\$freq" ] && echo "\$(basename \$cpu_dir) cur_freq  : \$(cat \$freq) kHz"
    done

    echo ""
    echo "--- NUMA auto-balancing ---"
    cat /proc/sys/kernel/numa_balancing 2>/dev/null \
        && echo "(0=disabled, 1=enabled)" \
        || echo "(not available)"

    echo ""
    echo "--- System load (at job start) ---"
    uptime
    echo "loadavg: \$(cat /proc/loadavg)"
    echo "nproc (visible CPUs): \$(nproc)"

    echo ""
    echo "--- Memory ---"
    free -h
    echo ""
    grep -E '^(MemTotal|MemFree|MemAvailable|HugePages_Total|Hugepagesize)' /proc/meminfo

    echo "========================================================"
} > "\$NODE_INFO_FILE" 2>&1

echo ">> node_info saved to: \$NODE_INFO_FILE"

# Uruchomienie programu
"$PROJECT_DIR/target/release/edupic" "${CYCLES}"

END_TIME=\$(date +"%Y-%m-%d %H:%M:%S")

echo "========================================================"
echo " End time : \$END_TIME"
echo "================================================="
EOF
