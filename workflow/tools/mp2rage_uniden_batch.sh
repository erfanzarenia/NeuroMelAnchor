#!/bin/bash
# Run MP2RAGE UNI denoising (mp2rage_uniden.py) over a BIDS directory.
#
# The MP2RAGE UNI image carries a strong background noise pattern outside the
# brain. mp2rage_uniden.py suppresses it by recombining UNI with the two
# inversion images; the regularisation factor trades background suppression
# against tissue contrast.
#
#   ./mp2rage_uniden_batch.sh --bids /path/to/bids
#   ./mp2rage_uniden_batch.sh --bids /path/to/bids --subjects sub-782
#
set -euo pipefail

bids_dir=""
subjects=""
regularisation=6
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_script="${script_dir}/mp2rage_uniden.py"

usage() {
    echo "Usage: $0 --bids DIR [--subjects 'sub-001 sub-002'] [--regularisation N]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids)           bids_dir="$2";       shift 2 ;;
        --subjects)       subjects="$2";       shift 2 ;;
        --regularisation) regularisation="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ -n "${bids_dir}" ]] || usage
[[ -d "${bids_dir}" ]] || { echo "Not a directory: ${bids_dir}" >&2; exit 1; }

# Default to every subject in the BIDS root.
if [[ -z "${subjects}" ]]; then
    subjects="$(cd "${bids_dir}" && ls -d sub-* 2>/dev/null || true)"
fi
[[ -n "${subjects}" ]] || { echo "No sub-* directories in ${bids_dir}" >&2; exit 1; }

for subject in ${subjects}; do
    sub_dir="${bids_dir}/${subject}"
    echo "Processing ${subject}..."

    inv1="${sub_dir}/anat/${subject}_acq-INV1.nii"
    inv2="${sub_dir}/anat/${subject}_acq-INV2.nii"
    uni="${sub_dir}/anat/${subject}_acq-MP2RAGEUNI_run-01_T1w.nii"
    output="${sub_dir}/anat/${subject}_desc-UNIDEN_run-01_T1w.nii"

    for required in "${inv1}" "${inv2}" "${uni}"; do
        if [[ ! -f "${required}" ]]; then
            echo "  skipping ${subject}: missing $(basename "${required}")" >&2
            continue 2
        fi
    done

    python "${python_script}" "${uni}" "${inv1}" "${inv2}" "${output}" "${regularisation}"
done

echo "All subjects processed."
