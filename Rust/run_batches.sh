#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET=1000
BATCH=100
STATE_FILE=".edupic_checkpoint"
LOG_FILE="run_batches.log"
BIN="./target/release/edupic"
MEASURE_LAST=false
MEASUREMENT_EVERY=0
MEASUREMENT_DIR="measurements"
FRESH=false
START_FROM=""
STATUS_ONLY=false

usage() {
	cat <<'EOF'
Usage:
  ./run_batches.sh [options]

Options:
  --target N            Target checkpoint to reach (default: 5000)
  --batch N             Number of cycles per batch (default: 100)
  --fresh               Start from scratch: remove old outputs and run init cycle (0)
  --start-from N        Trust that the current checkpoint is N and continue from there
  --measurement-every N Run measurement when a batch crosses each multiple of N cycles
  --measurement-last    Run the final batch with measurement mode: ... edupic STEP m
  --status              Print the script's saved checkpoint and exit
  -h, --help            Show this help

Notes:
  - The init run "edupic 0" creates checkpoint 1.
  - This script stores the last completed checkpoint in .edupic_checkpoint.
  - If interrupted, rerunning the script resumes from the last full batch recorded there.
  - Measurement files are copied to measurements/checkpoint_XXXXX/.
  - After --fresh, the init run creates checkpoint 1, so with --batch 100 and --measurement-every 500
	the first measurement is taken by the batch ending at checkpoint 501.
  - If picdata.bin exists but .edupic_checkpoint does not, use --start-from N or --fresh.
EOF
}

cleanup_outputs() {
	rm -f \
		picdata.bin conv.dat cs.dat info.txt density.dat eepf.dat efed.dat ifed.dat \
		pot_xt.dat efield_xt.dat ne_xt.dat ni_xt.dat je_xt.dat ji_xt.dat \
		powere_xt.dat poweri_xt.dat ioniz_xt.dat meanee_xt.dat meanei_xt.dat \
		plot_*.png "$STATE_FILE" "$LOG_FILE"
	rm -rf "$MEASUREMENT_DIR"
}

save_measurement_snapshot() {
	local checkpoint="$1"
	local out_dir
	out_dir=$(printf "%s/checkpoint_%05d" "$MEASUREMENT_DIR" "$checkpoint")
	mkdir -p "$out_dir"

	local files=(
		info.txt density.dat eepf.dat efed.dat ifed.dat
		pot_xt.dat efield_xt.dat ne_xt.dat ni_xt.dat je_xt.dat ji_xt.dat
		powere_xt.dat poweri_xt.dat ioniz_xt.dat meanee_xt.dat meanei_xt.dat
		conv.dat picdata.bin
	)

	local file
	for file in "${files[@]}"; do
		if [[ -f "$file" ]]; then
			cp -f "$file" "$out_dir/"
		fi
	done
}

require_integer() {
	local value="$1"
	local name="$2"
	if ! [[ "$value" =~ ^[0-9]+$ ]]; then
		echo "Błąd: $name musi być nieujemną liczbą całkowitą." >&2
		exit 1
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--target)
			TARGET="${2:-}"
			shift 2
			;;
		--batch)
			BATCH="${2:-}"
			shift 2
			;;
		--start-from)
			START_FROM="${2:-}"
			shift 2
			;;
		--measurement-every)
			MEASUREMENT_EVERY="${2:-}"
			shift 2
			;;
		--measurement-last)
			MEASURE_LAST=true
			shift
			;;
		--fresh)
			FRESH=true
			shift
			;;
		--status)
			STATUS_ONLY=true
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Nieznana opcja: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

require_integer "$TARGET" "--target"
require_integer "$BATCH" "--batch"
[[ -n "$START_FROM" ]] && require_integer "$START_FROM" "--start-from"
require_integer "$MEASUREMENT_EVERY" "--measurement-every"

if (( BATCH == 0 )); then
	echo "Błąd: --batch musi być > 0." >&2
	exit 1
fi

if (( MEASUREMENT_EVERY > 0 && TARGET < MEASUREMENT_EVERY )); then
	echo "Uwaga: --measurement-every jest większe niż --target, więc pomiary okresowe nie wystąpią." | tee -a "$LOG_FILE"
fi

if $STATUS_ONLY; then
	if [[ -f "$STATE_FILE" ]]; then
		echo "Ostatni checkpoint zapisany przez skrypt: $(<"$STATE_FILE")"
	else
		echo "Brak pliku $STATE_FILE. Skrypt nie zna aktualnego checkpointu."
	fi
	exit 0
fi

mkdir -p target/release

echo "[$(date '+%F %T')] Start run_batches.sh" | tee -a "$LOG_FILE"
echo "Target=$TARGET Batch=$BATCH Fresh=$FRESH MeasureEvery=$MEASUREMENT_EVERY MeasureLast=$MEASURE_LAST" | tee -a "$LOG_FILE"

cargo build --release | tee -a "$LOG_FILE"

mkdir -p "$MEASUREMENT_DIR"

current=0

if $FRESH; then
	echo "Czyszczenie poprzednich wyników..." | tee -a "$LOG_FILE"
	cleanup_outputs
	cargo build --release | tee -a "$LOG_FILE"
	echo "Uruchamiam cykl inicjalizacyjny: $BIN 0" | tee -a "$LOG_FILE"
	"$BIN" 0 | tee -a "$LOG_FILE"
	current=1
	echo "$current" > "$STATE_FILE"
elif [[ -n "$START_FROM" ]]; then
	current="$START_FROM"
	echo "$current" > "$STATE_FILE"
elif [[ -f "$STATE_FILE" ]]; then
	current="$(<"$STATE_FILE")"
elif [[ -f picdata.bin ]]; then
	echo "Wykryto picdata.bin, ale brak $STATE_FILE." >&2
	echo "Podaj --start-from N albo użyj --fresh." >&2
	exit 1
elif [[ ! -f picdata.bin ]]; then
	echo "Brak picdata.bin, uruchamiam cykl inicjalizacyjny: $BIN 0" | tee -a "$LOG_FILE"
	"$BIN" 0 | tee -a "$LOG_FILE"
	current=1
	echo "$current" > "$STATE_FILE"
fi

if (( current > TARGET )); then
	echo "Aktualny checkpoint ($current) jest już większy niż target ($TARGET)." | tee -a "$LOG_FILE"
	exit 0
fi

while (( current < TARGET )); do
	remaining=$(( TARGET - current ))
	step=$BATCH
	if (( remaining < step )); then
		step=$remaining
	fi

	cmd=("$BIN" "$step")
	next=$(( current + step ))
	take_measurement=false
	measurement_threshold=""
	if (( MEASUREMENT_EVERY > 0 )); then
		measurement_threshold=$(( ((current / MEASUREMENT_EVERY) + 1) * MEASUREMENT_EVERY ))
		if (( current < measurement_threshold && measurement_threshold <= next )); then
			take_measurement=true
		fi
	fi
	if $MEASURE_LAST && (( next >= TARGET )); then
		take_measurement=true
	fi

	if $take_measurement; then
		cmd+=("m")
	fi

	echo "[$(date '+%F %T')] Checkpoint=$current, uruchamiam batch=$step, cel po batchu=$next, measurement=$take_measurement, prog=${measurement_threshold:-brak}" | tee -a "$LOG_FILE"
	"${cmd[@]}" | tee -a "$LOG_FILE"

	current=$next
	echo "$current" > "$STATE_FILE"
	if $take_measurement; then
		save_measurement_snapshot "$current"
		echo "[$(date '+%F %T')] Zapisano snapshot pomiarowy w $MEASUREMENT_DIR/checkpoint_$(printf '%05d' "$current")" | tee -a "$LOG_FILE"
	fi
	echo "[$(date '+%F %T')] Zakończono batch. Nowy checkpoint=$current" | tee -a "$LOG_FILE"
done

echo "Gotowe. Osiągnięto checkpoint $current." | tee -a "$LOG_FILE"
