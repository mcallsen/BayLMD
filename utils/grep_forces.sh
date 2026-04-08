COUNT=$1
NATOMS=$2

for ((i = 1; i <= ${COUNT}; i++)) ; do
    echo Direct ${NATOMS} ${i} >> forces.dat
    grep -A $(( ${NATOMS} + 1)) "TOTAL-FORCE" output/OUTCAR_${i} | tail -n ${NATOMS} | gawk '{print $4, $5, $6}' >> forces.dat
done

