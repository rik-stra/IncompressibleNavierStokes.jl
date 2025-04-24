#!/bin/bash

for i in {2..5}
do
    julia --project track_ref.jl $i
    julia --project train_LinReg.jl $i
    julia --project online_sgs.jl $i
done
