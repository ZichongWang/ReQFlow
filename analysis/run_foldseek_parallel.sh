#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <pdb_list> <designable_list> <output_dir> <database> <result_summary>"
    echo "Example: $0 /path/to/pdb_list.txt /path/to/designable_list.txt /path/to/output_dir pdb /path/to/result_summary.csv"
    exit 1
fi


pdb_list=$1
designable_list=$2
output_dir=$3
database=$4
result_summary=$5

echo "PDB list: $pdb_list"
echo "Designable list: $designable_list"
echo "Output directory: $output_dir"
echo "Database: $database"
echo "Result summary: $result_summary"

# Make sure the output directory exists
mkdir -p "$output_dir"

# Write the header to the result summary file
echo -e "PDB File,Max TM-score,Designable" > "$result_summary"

tmp_result_dir="${output_dir}/parallel_results"
mkdir -p "$tmp_result_dir"

process_pdb() {
    local pdb_path="$1"
    local output_dir="$2"
    local database="$3"
    local tmp_result_dir="$4"
    local designable_list="$5"

    if [ ! -f "$pdb_path" ]; then
        sleep 0.1
        if [ ! -f "$pdb_path" ]; then
            echo "File not found: $pdb_path"
            exit 1
        fi
    fi

    file_name=$(basename "$pdb_path" .pdb)
    dir_name_1=$(basename "$(dirname "$pdb_path")")
    dir_name_2=$(basename "$(dirname "$(dirname "$pdb_path")")")

    pdb_name="${dir_name_2}_${dir_name_1}_${file_name}"
    aln_file="$output_dir/${pdb_name}_aln.txt"
    tmp_folder="$output_dir/tmp_${pdb_name}"

    echo "Checking file: $pdb_path"
    ls -l "$pdb_path"
    echo "Running foldseek with:"
    echo "  Input PDB: $pdb_path"
    echo "  Database: $database"
    echo "  Output alignment file: $aln_file"
    echo "  Temporary folder: $tmp_folder"

    foldseek easy-search "$pdb_path" "$database" "$aln_file" "$tmp_folder" \
        --alignment-type 1 \
        --exhaustive-search \
        --max-seqs 10000000000 \
        --tmscore-threshold 0.0 \
        --gpu 1\
        --format-output query,target,alntmscore,lddt,evalue

    if [ ! -f "$aln_file" ]; then
        echo "No alignment result for $pdb_path" >&2
        exit 0
    fi

    # Extract the maximum TM-score
    # According to the issue of FoldSeek mentioned in \url{https://github.com/steineggerlab/foldseek/issues/323}, we use the E-value column to report the TM-score. 
    max_tmscore=$(awk '{if(NR>1) print $5}' "$aln_file" | sort -nr | head -1)
    if [ -z "$max_tmscore" ]; then
        max_tmscore="N/A"
    fi

    if grep -xFq "$pdb_path" "$designable_list"; then
        designable=1
    else
        designable=0
    fi

    echo -e "$pdb_path,$max_tmscore,$designable" > "${tmp_result_dir}/${pdb_name}_result.csv"

    rm -rf "$tmp_folder"
    rm -f "$aln_file"
}

export -f process_pdb
export output_dir database tmp_result_dir designable_list

CPU_CORES=$(nproc)
JOBS=$((CPU_CORES * 5 / 10))
# Using 50% of the CPU cores for parallel processing
parallel --jobs "$JOBS" process_pdb {} "$output_dir" "$database" "$tmp_result_dir" "$designable_list" :::: "$pdb_list"

tail -n +1 "${tmp_result_dir}"/*_result.csv >> "$result_summary"

rm -rf "$tmp_result_dir"

echo "All PDB files processed. Results saved in $result_summary"