#!/bin/bash

for i in {1..2}
do
    julia --project train_LinReg.jl $i
    julia --project online_sgs.jl $i
done
