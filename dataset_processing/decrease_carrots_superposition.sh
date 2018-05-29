# carrots

# carrots group A
for i in `seq 1 24`
do
    echo directory: carrot_s$i
    python decrease_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images_superposition/group_a/carrot_a$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/decreased_images_superposition/group_a/$i \
	   -l 0 -n 10
done

# carrots group B
for i in `seq 1 24`
do
    echo directory: carrot_l$i
    python decrease_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/original_images_superposition/group_b/carrot_b$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/decreased_images_superposition/group_b/$i \
	   -l 1
done

# python decrease_dataset.py \
#        -id ../../dataset/carrots/images_for_training_mcnn/original_images_superposition/group_b/carrot_b27 \
#        -od ../../dataset/carrots/images_for_training_mcnn/decreased_images_superposition/group_b/27 \
#        -l 0 -n 4
