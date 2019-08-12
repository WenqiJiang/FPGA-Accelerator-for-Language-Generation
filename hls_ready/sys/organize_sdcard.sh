mv sd_card LG_HW_16192_opt
mkdir LG_HW_16192_opt/run
mkdir LG_HW_16192_opt/run/run
mv LG_HW_16192_opt/c-rnn.elf LG_HW_16192_opt/run/run/c-rnn.elf
cp -r ../../model LG_HW_16192_opt/
cp -r ../../datasets/ LG_HW_16192_opt/
cp -r ../correct_results/ LG_HW_16192_opt/run
