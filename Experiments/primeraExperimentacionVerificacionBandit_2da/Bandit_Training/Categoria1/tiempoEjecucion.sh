#!/bin/bash

# Pedir la ruta del archivo de Python
read -p "Ingresa la ruta del archivo de Python que deseas ejecutar: " archivo_python

# Verificar si el archivo existe
if [[ ! -f "$archivo_python" ]]; then
  echo "El archivo $archivo_python no existe."
  exit 1
fi

# Usar el comando 'time' para medir el tiempo de ejecución del script de Python
echo "Ejecutando el archivo de Python..."
/usr/bin/time -f "Tiempo de ejecución:\nReal: %E\nUser: %U\nSys: %S" sudo python2.7 "$archivo_python"
