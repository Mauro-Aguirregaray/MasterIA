import tree_hanoi
import hanoi_states


## TP

def best_first_search(problem: hanoi_states.ProblemHanoi, display: bool = False):
    """
    Realiza una búsqueda primero en profundidad para encontrar una solución a un problema de Hanoi y recuerda si ya
    paso por un estado e ignora seguir buscando en ese nodo para evitar recursividad.

    Parameters:
        problem (hanoi_states.ProblemHanoi): El problema de la Torre de Hanoi a resolver.
        display (bool, optional): Muestra un mensaje de cuantos caminos se expandieron y cuantos quedaron sin expandir.
                                  Por defecto es False.

    Returns:
        tree_hanoi.NodeHanoi: El nodo que contiene la solución encontrada.
    """

    frontier = ([tree_hanoi.NodeHanoi(problem.initial)])  # Creamos una cola LIFO con el nodo inicial

    explored = set()  # Este set nos permite ver si ya exploramos un estado para evitar repetir indefinidamente
    while len(frontier) > 0:
        node = frontier.pop()  # Extraemos el último nodo de la cola

        # Agregamos nodo al set. Esto evita guardar duplicados, porque set nunca tiene elementos repetidos, esto sirve
        # porque heredamos el método __eq__ en tree_hanoi.NodeHanoi de aima.Node
        explored.add(node.state)

        if problem.goal_test(node.state):  # Comprobamos si hemos alcanzado el estado objetivo
            if display:
                print(len(explored), "caminos se expandieron y", len(frontier), "caminos quedaron en la frontera")
            return node
        # Agregamos a la cola todos los nodos sucesores del nodo actual que no haya visitados
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)

    return None




