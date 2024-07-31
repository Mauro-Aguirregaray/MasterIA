import os

import tracemalloc
import time
from hanoi_states import StatesHanoi, ProblemHanoi
from tree_hanoi import NodeHanoi
from search import (  # Importa las funciones de búsqueda del módulo search
    breadth_first_tree_search,
    breadth_first_graph_search,
    deep_first_graph_search
)

from simulator import  constants,logic,animator,simulation_hanoi
import numpy as np



def main(algorithm, disks = 5):
    
    """
    Función principal que resuelve el problema de la Torre de Hanoi y genera los JSON para el simulador.
    """
    # Define el estado inicial y el estado objetivo del problema
    initial_state = StatesHanoi([x for x in range(disks,0,-1) ], [], [], max_disks = disks)
    goal_state = StatesHanoi([], [], [x for x in range(disks,0,-1) ], max_disks= disks)

    # Crea una instancia del problema de la Torre de Hanoi
    problem_hanoi = ProblemHanoi(initial=initial_state, goal=goal_state)

    # Para medir tiempo consumido
    start_time = time.perf_counter()
    # Para medir memoria consumida (usamos el pico de memoria)
    tracemalloc.start()

    # Métodos no informados

    # Resuelve el problema utilizando búsqueda en anchura
    # Esta forma de búsqueda es muy ineficiente, por lo que si deseas probarlo, usa 3 discos o si querés esperar
    # un poco más, 4 discos, pero 5 discos no finaliza nunca.
    #last_node = breadth_first_tree_search(problem_hanoi)

    # Resuelve el problema utilizando búsqueda en anchura, pero con memoria que recuerda caminos ya recorridos.
    last_node = algorithm(problem_hanoi, display=False)

    _, memory_peak = tracemalloc.get_traced_memory()
    memory_peak /= 1024*1024
    tracemalloc.stop()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if isinstance(last_node, NodeHanoi):
        # Imprime la longitud del camino de la solución encontrada
       # print(f'Longitud del camino de la solución: {last_node.state.accumulated_cost}')

        # Genera los JSON para el simulador
        last_node.generate_solution_for_simulator()

    else:
        print(last_node)
        print("No se encuentra solución")

    # Imprime las métricas medidas
    #print(f"Tiempo que demoró: {elapsed_time} [s]", )
    #print(f"Maxima memoria ocupada: {round(memory_peak, 2)} [MB]", )

    return (elapsed_time,memory_peak,last_node.state.accumulated_cost)


def cleanEnviorment():
    initial_state_file = "initial_state.json"
    if os.path.exists(initial_state_file):
        os.remove(initial_state_file)

    sequence_file = "sequence.json"
    if os.path.exists(sequence_file):
        os.remove(sequence_file)

def isEnviromentReady():
    initial_state_file = "initial_state.json"
    sequence_file = "sequence.json"
    
    return os.path.exists(initial_state_file) and os.path.exists(sequence_file)

# Sección de ejecución del programa
if __name__ == "__main__":
    cleanEnviorment()

    runSimulator = True
    disks = 5

    memory = np.array([])
    timeProcess = np.array([])
    cost = np.array([])

    for x in range(1,11):
        #(elapsed_time,memory_peak,last_node.state.accumulated_cost)
        statistics = main(deep_first_graph_search,disks)

        timeProcess = np.append(timeProcess, statistics[0])
        memory = np.append(memory, statistics[1])
        cost = np.append(cost, statistics[2])


    print("Array tiempo",timeProcess)
    print("Array memory",memory)
    print("Array costo",cost)

    #A nivel implementación, ¿qué tiempo y memoria ocupa el algoritmo? (Se recomienda correr 10 veces y calcular promedio y desvío estándar de las métricas).
    timeProcessMean = np.mean(timeProcess)
    timeStandar = np.std(timeProcess)
    
    print (f"¿Qué tiempo ocupa el algoritmo? - Promedio: {timeProcessMean} segundos")
    print (f"¿Qué tiempo ocupa el algoritmo? - Desvío estándar: {timeStandar} segundos")


    memoryMean = np.mean(memory)
    memoryStandar = np.std(memory)
    
    print (f"¿Qué memoria ocupa el algoritmo? - Promedio: {memoryMean} Mb")
    print (f"¿Qué memoria ocupa el algoritmo? - Desvío estándar: {memoryStandar} Mb")

    print ('''#Si la solución óptima es $2^k - 1$ movimientos con *k* igual al número de discos. 
    #Qué tan lejos está la solución del algoritmo implementado de esta solución óptima (se recomienda correr al menos 10 veces y usar el promedio de trayecto usado).''')
    optimalSolution = (2 ** disks) - 1
    print ("Solución óptima", optimalSolution)


    costMean = np.mean(cost)
    print ("Costo (movimientos) promedio", costMean)

    print (f"Tenemos un costo extra de {costMean - optimalSolution} movimientos respecto a la solución óptima")

    if runSimulator:
        simulation_hanoi.main()
