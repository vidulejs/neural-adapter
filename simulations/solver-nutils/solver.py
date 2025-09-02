# Original source: https://github.com/evalf/nutils/blob/d73749ff7d64c9ccafdbb88cd442f80b9448c118/examples/burgers.py

from nutils import mesh, function, export, testing
from nutils.solver import System
from nutils.expression_v2 import Namespace
import treelog as log
import numpy
import itertools
import precice
import json
import os
import argparse

def main(dim: int,
         btype: str = 'discont',
         degree: int = 1,
         newtontol: float = 1e-5,
         config_file: str = "precice-config.xml"):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "..", "datagen", "config.json"), 'r') as f:
        config = json.load(f)["solver"]
    
    config_path = os.path.join(script_dir, "..", f"{dim}d", config_file)
    participant = precice.Participant("Solver", config_path, 0, 1)
    
    mesh_internal_name = f"Solver-Mesh-{dim}D-Internal"
    mesh_boundaries_name = f"Solver-Mesh-{dim}D-Boundaries"
    data_name = f"Data_{dim}D"
    ic_name = f"Initial_Condition_{dim}D"
    res = config[f"{dim}d_resolution"]
    domain_min = config[f"{dim}d_domain_min"]
    domain_max = config[f"{dim}d_domain_max"]
    nelems = res[0] - 1

    domain, geom = mesh.line(numpy.linspace(domain_min[0], domain_max[0], nelems+1), periodic=True)
    
    internal_coords = numpy.array([numpy.linspace(domain_min[0], domain_max[0], res[0]), numpy.full(res[0], domain_min[1])]).T
    boundary_coords = numpy.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_max[1]]])
    
    internal_vertex_ids = participant.set_mesh_vertices(mesh_internal_name, internal_coords)
    boundary_vertex_ids = participant.set_mesh_vertices(mesh_boundaries_name, boundary_coords)

    sample = domain.locate(geom, internal_coords[:, 0], tol=1e-5)
    
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.u = domain.field('u', btype=btype, degree=degree)
    ns.du = ns.u - function.replace_arguments(ns.u, 'u:u0')
    ns.v = domain.field('v', btype=btype, degree=degree)
    ns.t = function.field('t')
    ns.dt = ns.t - function.field('t0')
    ns.f = '.5 u^2'
    ns.C = 1

    res_pde = domain.integral('(v du / dt - ∇(v) f) dV' @ ns, degree=degree*2)
    res_pde -= domain.interfaces.integral('[v] n ({f} - .5 C [u] n) dS' @ ns, degree=degree*2)
    system = System(res_pde, trial='u', test='v')

    if not participant.requires_initial_data():
        participant.initialize()
        initial_condition_values = participant.read_data(mesh_internal_name, ic_name, internal_vertex_ids, 0.0)
        
        ns.uic = domain.basis('spline', degree=2).dot(initial_condition_values[:-1])

        sqr = domain.integral('(u - uic)^2 dV' @ ns, degree=max(degree*2, 5))
        args = System(sqr, trial='u').solve()
    else:
        participant.initialize()
        args = {}

    args['t'] = 0.

    with log.iter.plain('timestep', itertools.count()) as steps:
        for _ in steps:
            if not participant.is_coupling_ongoing():
                break

            timestep = participant.get_max_time_step_size()

            args = system.step(timestep=timestep, arguments=args, timearg='t', suffix='0', tol=newtontol)

            internal_data_values = sample.eval(ns.u, arguments=args)
            boundary_indices_in_internal_mesh = [numpy.where((internal_coords == b).all(axis=1))[0][0] for b in boundary_coords]
            boundary_data_values = internal_data_values[boundary_indices_in_internal_mesh]
            participant.write_data(mesh_internal_name, data_name, internal_vertex_ids, internal_data_values)
            participant.write_data(mesh_boundaries_name, data_name, boundary_vertex_ids, boundary_data_values)

            participant.advance(timestep)

    participant.finalize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, choices=[1, 2], help="Dimension of the simulation")
    parser.add_argument('--config_file', type=str, default="precice-config.xml")
    parser.add_argument('--btype', type=str, default='discont')
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--newtontol', type=float, default=1e-5)
    
    args_cli = parser.parse_args()

    main(dim=args_cli.dim, btype=args_cli.btype, degree=args_cli.degree, newtontol=args_cli.newtontol, config_file=args_cli.config_file)