# carrots

# carrots S size
for i in `seq 1 12`
do
    echo directory: carrot_s$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images_square/s_size/carrot_s$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images_square/s_size/$i \
	   -l 0
done

# carrots L size
for i in `seq 1 12`
do
    echo directory: carrot_l$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images_square/l_size/carrot_l$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images_square/l_size/$i \
	   -l 1
done
