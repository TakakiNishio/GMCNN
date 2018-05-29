# carrots

# carrots group A
for i in `seq 1 28`
do
    echo directory: carrot_s$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/decreased_images_superposition/group_a/$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images_superposition_from_few_images/group_a/$i \
	   -l 0
done

# carrots group B
for i in `seq 1 28`
do
    echo directory: carrot_l$i
    python increase_dataset.py \
	   -id ../../dataset/carrots/images_for_training_mcnn/decreased_images_superposition/group_b/$i \
	   -od ../../dataset/carrots/images_for_training_mcnn/increased_images_superposition_from_few_images/group_b/$i \
	   -l 1
done
