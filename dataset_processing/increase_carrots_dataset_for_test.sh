# carrots

echo directory: negative
python increase_dataset_for_test.py -itd data_test/negative \
       -oid ../../dataset/carrots/images_for_test/increased_images/test_images_negative/ \
       -otd data_test_increased/negative

echo directory: positive
python increase_dataset_for_test.py -itd data_test/positive \
       -oid ../../dataset/carrots/images_for_test/increased_images/test_images_positive/ \
       -otd data_test_increased/positive

echo directory: mixed
python increase_dataset_for_test.py -itd data_test/mixed \
       -oid ../../dataset/carrots/images_for_test/increased_images/test_images_mixed/ \
       -otd data_test_increased/mixed

