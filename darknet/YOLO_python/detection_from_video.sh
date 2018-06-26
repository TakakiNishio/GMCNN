m="15"
l="carrot"

# carrots group_a
for i in `seq 1 45`
do
    echo video: carrot_a$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/group_a/carrot_a$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/from_videos/group_a/carrot_a$i -m $m
    sleep 5
done

# carrots group_b
for i in `seq 1 45`
do
    echo video: carrot_b$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/group_b/carrot_s$i.avi -l $l -od ../../../dataset/carrots/detected_by_gcnn/from_videos/group_b/carrot_b$i -m $m
    sleep 5
done

