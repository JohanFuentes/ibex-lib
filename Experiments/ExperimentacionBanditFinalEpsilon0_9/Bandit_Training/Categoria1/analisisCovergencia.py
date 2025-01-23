def procesar_archivo(archivo):
    # Inicializa las listas de conteo para primeras y segundas líneas
    conteo_primeras = [0] * 8
    conteo_segundas = [0] * 8

    # Listas para almacenar las líneas de salida
    output_primeras = []
    output_segundas = []

    with open(archivo, 'r') as file:
        pares = []
        for line in file:
            line = line.strip()
            if line:  # Ignora líneas en blanco
                pares.append(list(map(int, line.split())))
                if len(pares) == 2:
                    # Procesa el par de líneas
                    primera_linea, segunda_linea = pares
                    max_primera = max(primera_linea)
                    max_segunda = max(segunda_linea)

                    # Genera salida para primera línea del par
                    primera_salida = [1 if x == max_primera else 0 for x in primera_linea]
                    output_primeras.append(primera_salida)
                    # Cuenta los índices máximos en primera línea
                    for j, val in enumerate(primera_linea):
                        if val == max_primera:
                            conteo_primeras[j] += 1

                    # Genera salida para segunda línea del par
                    segunda_salida = [1 if x == max_segunda else 0 for x in segunda_linea]
                    output_segundas.append(segunda_salida)
                    # Cuenta los índices máximos en segunda línea
                    for j, val in enumerate(segunda_linea):
                        if val == max_segunda:
                            conteo_segundas[j] += 1

                    # Resetea pares para el siguiente par de líneas
                    pares = []

    # Imprime la salida
    print("Primeras Lineas:")
    for line in output_primeras:
        print(" ".join(map(str, line)))
    print("\nConteo Primeras Lineas:")
    print(" ".join(map(str, conteo_primeras)))

    print("\nSegundas Lineas:")
    for line in output_segundas:
        print(" ".join(map(str, line)))
    print("\nConteo Segundas Lineas:")
    print(" ".join(map(str, conteo_segundas)))


# Uso de la función
procesar_archivo('Logs.txt')
