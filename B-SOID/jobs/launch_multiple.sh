#!/bin/bash

for i in {1..20}
do
    bash ./2phase/cluster_collect_embed run-$i
done