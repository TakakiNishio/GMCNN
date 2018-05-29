# carrots

for i in `seq 1 12`
do
    echo directory: carrot_s$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images/s_size/carrot_s$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images/s_size/$i \
	   -l 0
done


for i in `seq 1 12`
do
    echo directory: carrot_l$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images/l_size/carrot_l$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images/l_size/$i \
	   -l 1
done
