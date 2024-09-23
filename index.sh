#!/bin/bash

jacutingaites=($(ls -p "Parameters" | grep -v /))

for i in "${!jacutingaites[@]}"; do
    echo "$((i+1))) ${jacutingaites[$i]}"
done

read -p "Choose one of the listed jacuntigaites.: " choose

if [ "$choose" -gt 0 ] && [ "$choose" -le ${#jacutingaites[@]} ]; then
    jacutingaite="${jacutingaites[$((choose-1))]}"
    echo "Selected: $jacutingaite"
else
    echo "Invalid option. Exit."
    exit 1
fi

echo "from Parameters.${jacutingaite::-3} import *" > "MNX.py"

PS3="Crystalline cell type: "

optionsC=("primitive" "rectangular")

select choose in "${optionsC[@]}"; do
    case $REPLY in
        1)
            echo "Primitive cell"
            type_cell="${optionsC[0]}"
            break
            ;;
        2)
            echo "Rectangular cell"
            type_cell="${optionsC[1]}"
            break
            ;;
        *)
            echo "Invalid option. Try again."
            ;;
    esac
done

PS3="Consider spin-orbit coupling: "

optionsS=("True" "False")

select choose in "${optionsS[@]}"; do
    case $REPLY in
        1)
            echo "Yes"
            w_soc="${optionsS[0]}"
            break
            ;;
        2)
            echo "No"
            w_soc="${optionsS[1]}"
            break
            ;;
        *)
            echo "Invalid option. Try again."
            ;;
    esac
done

python3 index.py $type_cell $w_soc