for i in $(seq $1 $3 $2) 
do 
    echo Direct $i 256
    tail -n 256 $i/POSCAR
done > structures.vasp
