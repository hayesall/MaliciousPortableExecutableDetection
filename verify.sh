#!/bin/bash

if [[ ! -d "exe-dir/" ]]; then
    echo "Error, exe-dir not found."
    exit
fi

if [[ ! -e "classifier/classifier.pkl" ]]; then
    echo "Error, couldn't find the classifier."
    exit
fi

if [[ ! -e "classifier/features.pkl" ]]; then
    echo "Error, couldn't find the features."
    exit
fi

i=1
for file in exe-dir/*; do
    OUTPUT=$(python checkfile.py $file)
    FILE=$(echo "$OUTPUT" | cut -d ' ' -f 1)
    BELIEF=$(echo "$OUTPUT" | cut -d ' ' -f 3)
    printf "%-2s. %-45s %-10s\n" "$i" "$FILE" "$BELIEF"
    i=$((i+1))
done
