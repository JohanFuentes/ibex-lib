#!/bin/bash

# Pedir la ruta del archivo de Python
read -p "Ingresa la ruta del archivo de Python que deseas ejecutar: " archivo_python

# Verificar si el archivo existe
if [[ ! -f "$archivo_python" ]]; then
  echo "El archivo $archivo_python no existe."
  exit 1
fi

# Pedir el número de veces que se desea ejecutar el archivo
read -p "¿Cuántas veces deseas ejecutar el archivo? " num_ejecuciones

# Verificar si el número de ejecuciones es un número entero válido
if ! [[ "$num_ejecuciones" =~ ^[0-9]+$ ]]; then
  echo "Por favor, ingresa un número entero válido."
  exit 1
fi

# Ejecutar el archivo de Python el número de veces indicado
for ((i=1; i<=num_ejecuciones; i++))
do
  echo "Ejecución #$i"
  sudo python2.7 "$archivo_python"

  # Si no es la última iteración, eliminar los archivos y la carpeta
  if [[ $i -lt $num_ejecuciones ]]; then
    echo "Eliminando archivos y carpeta después de la ejecución #$i..."
    
    # Eliminar archivo state.out
    sudo rm -f state.out
    
    # Eliminar archivo final_results.txt
    sudo rm -f final_results.txt

    # Eliminar carpeta bandit_training_c1
    sudo rm -rf bandit_training_c1
    
    echo "Archivos eliminados."
  fi
done

echo "El script ha sido ejecutado $num_ejecuciones veces."
