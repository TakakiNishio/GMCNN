m="15"
l="carrot"

# carrots L size
for i in `seq 1 12`
do
    echo video: carrot_l$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/l_size/carrot_l$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/from_videos/l_size/carrot_l$i -m $m
    sleep 5
done

# carrots S size
for i in `seq 1 12`
do
    echo video: carrot_s$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/s_size/carrot_s$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/from_videos/s_size/carrot_s$i -m $m
    sleep 5
done

