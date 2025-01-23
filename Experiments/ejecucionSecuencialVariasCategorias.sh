#!/bin/bash

# Ejecuta el primer script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionBanditFinal/Bandit_Training/Categoria1/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Primer script ejecutado correctamente."
else
    echo "Error en el primer script." >&2
    exit 1
fi

# Ejecuta el segundo script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionBanditFinal/Bandit_Training/Categoria2/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Segundo script ejecutado correctamente."
else
    echo "Error en el segundo script." >&2
    exit 1
fi

# Ejecuta el tercer script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionBanditFinal/Bandit_Training/Categoria3/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Tercer script ejecutado correctamente."
else
    echo "Error en el tercer script." >&2
    exit 1
fi

# Ejecuta el cuarto script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionSarsaFinal/Sarsa_Training/Categoria1/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Cuarto script ejecutado correctamente."
else
    echo "Error en el cuarto script." >&2
    exit 1
fi

# Ejecuta el quinto script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionSarsaFinal/Sarsa_Training/Categoria2/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Quinto script ejecutado correctamente."
else
    echo "Error en el quinto script." >&2
    exit 1
fi

# Ejecuta el sexto script de Python
sudo python2.7 /home/johan/MAGISTER/ImplementacionCodigo/ibex-lib/Experiments/ExperimentacionSarsaFinal/Sarsa_Training/Categoria3/fwk4xps.py
if [ $? -eq 0 ]; then
    echo "Sexto script ejecutado correctamente."
else
    echo "Error en el sexto script." >&2
    exit 1
fi
