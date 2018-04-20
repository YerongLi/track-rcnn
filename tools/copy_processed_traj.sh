#!/bin/bash
train_videos=(  "TUD-Stadtmitte"
                "TUD-Campus"
                "PETS09-S2L1"
                "ETH-Bahnhof"
                "ETH-Sunnyday"
                "ETH-Pedcross2"
                "ADL-Rundle-6"
                "ADL-Rundle-8"
                "KITTI-13"
                "KITTI-17"
                "Venice-2")
src_path="/cvgl/u/kuanfang/Datasets/processed"
dst_path="./data/2DMOT2015/train"

for name in "${train_videos[@]}"
do
    echo $name
    cp -r $src_path/$name/traj/* $dst_path/$name/traj/
done
