#!/bin/bash

# Asegúrate de que el script reciba dos argumentos: el comando a ejecutar y el número de veces
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <comando> <veces>"
    exit 1
fi

# Guardar los argumentos en variables
comando=$1
veces=$2

# Bucle para ejecutar el comando la cantidad de veces especificada
for ((i=1; i<=veces; i++))
do
    echo "Ejecución $i: $comando"
    eval $comando
done