#!/bin/bash
for i in {0..7}
do
     convert 3d_${i}.png -crop 3500x1800+1950+2650 3d_${i}_cropped.png
done

