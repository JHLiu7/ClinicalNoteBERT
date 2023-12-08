#!/bin/bash

NTYPE=radiology

for seq in sentence segment note
do
    cat ${RAW_DIR}/${NTYPE}/${seq}/*.txt > ${RAW_DIR}/${NTYPE}/${seq}.txt
done 

echo concated $NTYPE
