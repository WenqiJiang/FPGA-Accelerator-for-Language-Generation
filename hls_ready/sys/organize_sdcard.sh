mv sd_card LG_HW_16192
mkdir LG_HW_16192/run
mkdir LG_HW_16192/run/run
mv LG_HW_16192/c-rnn.elf LG_HW_16192/run/run/c-rnn.elf
cp -r ../../model LG_HW_16192/
cp -r ../../datasets/ LG_HW_16192/
cp -r ../correct_results/ LG_HW_16192/run
