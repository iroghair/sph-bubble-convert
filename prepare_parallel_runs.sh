#!/bin/bash

# Prepare parallel simulation re-runs from existing dump files.
# Each consecutive pair of dump files defines a sub-interval.
# A subfolder is created for each interval with adjusted GLS3D.DAT.

BASEDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASEDIR"

SIMULATOR="GLS3D_rev7_beta9_nospline_icc"
DATFILE="GLS3D.DAT"

# Collect dump file numbers (only files matching F[1-9]????*.ft3, i.e. >= 10000)
# Change the pattern according to the actual dump files present in the directory
dumps=()
for f in F[1-9]{0,2,4,6,8}???*.ft3; do
    num="${f#F}"
    num="${num%.ft3}"
    dumps+=("$num")
done

# Sort numerically
IFS=$'\n' sorted=($(sort -n <<<"${dumps[*]}")); unset IFS

echo "Found ${#sorted[@]} dump files"
echo "Sorted dump numbers: ${sorted[*]}"
echo ""

# Create sub-intervals from consecutive pairs
for ((i = 0; i < ${#sorted[@]} - 1; i++)); do
    curr="${sorted[$i]}"
    next="${sorted[$i+1]}"

    # Total_time = next_number / 1e4, formatted to match DAT file style
    total_time=$(awk "BEGIN {printf \"%.14E\", $next / 10000.0}")

    dumpfile="F${curr}.ft3"
    subdir="run_${curr}_to_${next}"

    echo "Creating $subdir (restart from $dumpfile, Total_time=$total_time)"

    # Create subfolder and output directory
    mkdir -p "$subdir/output"

    # Copy simulator, dump file, settings, and conversion script
    cp "$SIMULATOR" "$subdir/"
    cp "$dumpfile" "$subdir/"
    cp "$DATFILE" "$subdir/"
    cp "script.sh" "$subdir/"

    # Patch script.sh to point to the converter using absolute path
    sed -i "s|^CONVERTER=.*|CONVERTER=\"${BASEDIR}/bin/converter\"|" "$subdir/script.sh"

    # Adjust Total_time in the subfolder's GLS3D.DAT
    sed -i "s/^Total_time  : .*/Total_time  : ${total_time}/" "$subdir/$DATFILE"

    # Change dump frequency from 2000 to 10
    sed -i "s/^fd     : .*/fd     : 10/" "$subdir/$DATFILE"
done

echo ""
echo "Created $((${#sorted[@]} - 1)) sub-folders."
echo ""
echo "To run all in parallel:"
echo '  for dir in run_*_to_*; do'
echo '    (cd "$dir" && bash script.sh > script.log 2>&1 &)'
echo '    (cd "$dir" && ./'"$SIMULATOR"' F*.ft3 > output.txt 2>&1 &)'
echo '  done'
echo '  wait'
