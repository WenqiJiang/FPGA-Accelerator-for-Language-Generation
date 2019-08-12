mv sd_card LG_HW_6144
mkdir LG_HW_6144/run
mkdir LG_HW_6144/run/run
mv LG_HW_6144/c-rnn.elf LG_HW_6144/run/run/c-rnn.elf
cp -r ../../model LG_HW_6144/
cp -r ../../datasets/ LG_HW_6144/
cp -r ../correct_results/ LG_HW_6144/run
