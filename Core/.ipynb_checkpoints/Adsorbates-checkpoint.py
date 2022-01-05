from Core.Nanoparticle import Nanoparticle
import Core.MathModules as math

from ase.cluster import Octahedron
from ase import Atoms

import numpy as np

class Adsorbates(Nanoparticle):
    def __init__(self):
        Nanoparticle.__init__(self)
        
    def addatom_ontop(self, indices, distance):
        atoms_addatoms = self.get_ase_atoms()
        
        for index in indices:
            cn = self.get_coordination_number(index)
            position = self.get_position(index)
            unit, length = math.get_unit_vector(position)
            tilted_vector, _ = math.get_unit_vector(unit + math.get_perpendicular_vector(unit))
        
            if cn == 4 or cn == 6:
                C_distance = unit*(length + distance)
                O_distance = C_distance + (tilted_vector*1.15)

            if cn == 7:
                edge_perp_vec = math.get_perpendicular_edge_vector(position)
                perp_vector = math.get_perpendicular_vector(edge_perp_vec)
                tilted_vector,_  = math.get_unit_vector(edge_perp_vec + perp_vector)                                   
                C_distance = (unit * length) + (edge_perp_vec * distance)
                O_distance = C_distance + (tilted_vector*1.15)

            if cn == 9:
                around = self.get_coordination_atoms(index)
                plane = [self.get_position(x) for x in around if self.get_coordination_number(x) < 12]
                normal = math.get_normal_vector(plane)
                dot_prod = math.get_dot_product(unit, normal)
                direction = dot_prod/abs(dot_prod)
                normal = normal * direction
                perp_vector = math.get_perpendicular_vector(normal)
                tilted_vector, _ = math.get_unit_vector(normal + perp_vector)
                C_distance = (unit*length) + (normal*distance)
                O_distance = C_distance + (tilted_vector*1.15)

            add_atom1 = Atoms('O')
            #add_atom2 = Atoms('O')    
            add_atom1.translate(C_distance)
            #add_atom2.translate(O_distance)
            atoms_addatoms += add_atom1
            #atoms_addatoms += add_atom2
        
        return atoms_addatoms
    
    def find_plane_np(self, tolerance = 1):
        indices = self.get_atoms_in_the_surface_plane(7, edges_corner=True)
        normal_vector, d = math.get_plane([self.get_position(x) for x in indices[:3]])
    
        plane = []
        for indx in self.get_indices():
            dot_prod = np.dot(self.get_position(indx), normal_vector)
            if dot_prod > d - tolerance and dot_prod < d + tolerance:
                plane.append(indx)
                
        return plane, normal_vector
    
    def find_bridge_positions(self):
        bridge_positions = []
        plane = set(self.find_plane_np()[0])
        for central_atom_idex in plane:
            plane_nearest_neighbors = set(self.get_coordination_atoms(central_atom_idex))
            for nearest_neighbor in plane.intersection(plane_nearest_neighbors):
                bridge_positions.append([central_atom_idex, nearest_neighbor])
        return bridge_positions
    
    def add_atom_bridge(self, tolerance, add_atom, bridge_positions):
        plane, normal_vector = self.find_plane_np(tolerance)
        particle_add = self.get_ase_atoms()
    
        for bridge_position in bridge_positions:
            position_1 = self.get_position(bridge_position[0])
            position_2 = self.get_position(bridge_position[1])
            point = math.find_middle_point(position_1 , position_2) 
            unit_vector1, length1 = math.get_unit_vector(point)
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            ads = Atoms(add_atom)
            bridge_position = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
            ads.translate(bridge_position)
            particle_add += ads

        return particle_add
