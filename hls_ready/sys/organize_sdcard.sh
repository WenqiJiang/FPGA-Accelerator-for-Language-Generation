mv sd_card LG_HW_4096
mkdir LG_HW_4096/run
mkdir LG_HW_4096/run/run
mv LG_HW_4096/c-rnn.elf LG_HW_4096/run/run/c-rnn.elf
cp -r ../../model LG_HW_4096/
cp -r ../../datasets/ LG_HW_4096/
cp -r ../correct_results/ LG_HW_4096/run
