#!/bin/bash

for i in {1..20}
do
    bash ./2phase/bsoid_stability run-$i
done