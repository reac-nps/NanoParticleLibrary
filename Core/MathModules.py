import numpy as np

def get_atom_position(atom):
    return atom.position

def get_dot_product(v1, v2):
    return np.dot(v1,v2)

def get_unit_vector(position):
    length = np.linalg.norm(position)
    unit_vector = position / length

    return unit_vector, length

def get_perpendicular_vector(v, tolerance = 1e-04):
    if abs(v[1]) < tolerance and abs(v[2]) < tolerance:
        if abs(v[0]) < tolerance:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0 , 0])

def find_middle_point(positions):
    return sum(positions)/len(positions)    

def get_perpendicular_edge_vector(position, tolerance=1e-4):
    final_vector = np.array([0,0,0])

    for i, coordinate in enumerate(position):
        trial_vector = np.array([0,0,0])
        if abs(coordinate) > tolerance:
            trial_vector[i] = coordinate/abs(coordinate)
            final_vector += trial_vector
        else:
            final_vector += trial_vector
    final_vector, _ = get_unit_vector(final_vector)
    return final_vector

def get_normal_vector(positions):
    v1 = positions[1] - positions[0]
    v2 = positions[2] - positions[0]
    normal = np.cross(v1, v2)
    unit, _ = get_unit_vector(normal)
    return unit

def get_plane(positions):
    normal_vector = get_normal_vector(positions)
    d = np.dot(normal_vector, positions[2])
    return normal_vector, d

def get_bridge_perpendicular_line(positions, center_of_mass):
    a, a1 = positions
    n,_ = get_unit_vector(a - a1)
    p = center_of_mass
    direction = (p - a) - (np.dot((p-a), n)) * n
    return direction / np.linalg.norm(direction)

    
