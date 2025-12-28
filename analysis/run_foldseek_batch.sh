#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <pdb_list> <designable_list> <output_dir> <database> <result_summary>"
    exit 1
fi

pdb_list=$1
designable_list=$2
output_dir=$3
database=$4
result_summary=$5

# Resource Management
CPU_CORES=$(nproc)
THREADS=$((CPU_CORES * 5 / 10)) # Using 50% as per your original script

echo "Initializing optimized Batch Foldseek..."
mkdir -p "$output_dir"
tmp_dir="${output_dir}/tmp_batch"
mkdir -p "$tmp_dir"

# 1. Create the Query Database
# This encodes all PDBs into the 3Di alphabet in one parallel step.
# We use a trick to ensure 'query' names in the DB match the base file names
echo "Creating query database from list..."
foldseek createdb "$pdb_list" "${output_dir}/queryDB" --allow-id-transfer

# 2. Perform a single Batch Search
# This avoids reloading the 'database' index for every single query.
# We keep your exact flags: --alignment-type 1 and --exhaustive-search.
echo "Running structural search (GPU accelerated)..."
foldseek search "${output_dir}/queryDB" "$database" "${output_dir}/alnDB" "$tmp_dir" \
    --threads "$THREADS" \
    --alignment-type 1 \
    --exhaustive-search \
    --max-seqs 10000 \
    --tmscore-threshold 0.0 \

# 3. Convert results to TSV
# We extract query, target, and alntmscore. 
echo "Extracting alignment metrics..."
foldseek convertalis "${output_dir}/queryDB" "$database" "${output_dir}/alnDB" "${output_dir}/raw_results.tsv" \
    --format-output "query,target,alntmscore"

# 4. Final Processing and CSV Generation
# This AWK script performs a 3-way join:
#   - Map 1: Basename -> Full Path (from your original pdb_list)
#   - Map 2: Designable Status (from your designable_list)
#   - Data: Max TM-score from results
echo "Generating final summary..."
awk -v pdb_list="$pdb_list" -v designable_list="$designable_list" '
    BEGIN {
        FS="\t"; OFS=",";
        # Step A: Load Full Path mapping
        # Foldseek query IDs are usually basenames. We map them back to paths.
        while ((getline < pdb_list) > 0) {
            full_path = $0;
            n = split(full_path, parts, "/");
            basename = parts[n];
            sub(/\.pdb$/, "", basename); # Remove extension
            path_map[basename] = full_path;
            # Initialize TM-score for all files in list to 0 or N/A
            max_tm[full_path] = "0.0"; 
        }
        # Step B: Load Designable list
        while ((getline < designable_list) > 0) {
            is_designable[$0] = 1;
        }
        print "PDB File,Max TM-score,Designable";
    }
    # Step C: Process Foldseek results
    {
        q_name = $1; tm_val = $3;
        full_p = path_map[q_name];
        if (full_p != "" && tm_val > max_tm[full_p]) {
            max_tm[full_p] = tm_val;
        }
    }
    END {
        # Loop through path_map to ensure EVERY input PDB is in output, 
        # even if it had no hits in the database.
        for (base in path_map) {
            p = path_map[base];
            d = (is_designable[p] == 1) ? 1 : 0;
            # Match your original "N/A" logic if score is 0
            score = (max_tm[p] == "0.0") ? "N/A" : max_tm[p];
            print p, score, d;
        }
    }
' "${output_dir}/raw_results.tsv" > "$result_summary"

# Cleanup
echo "Cleaning up temporary database files..."
rm -rf "${output_dir}/queryDB"* "${output_dir}/alnDB"* "$tmp_dir" "${output_dir}/raw_results.tsv"

echo "Done. Results saved to $result_summary"