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

def main(btype: str = 'discont',
         degree: int = 1,
         newtontol: float = 1e-5):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "..", "datagen", "config.json"), 'r') as f:
        config = json.load(f)["solver"]
    
    config_path = os.path.join(script_dir, "..", "1d", "precice-config.xml")
    participant = precice.Participant("Solver", config_path, 0, 1)
    
    # --- Nutils Domain and preCICE Mesh Setup ---
    mesh_internal_name = "Solver-Mesh-1D-Internal"
    mesh_boundaries_name = "Solver-Mesh-1D-Boundaries"
    data_name = "Data_1D"
    res = config["1d_resolution"]
    domain_min = config["1d_domain_min"]
    domain_max = config["1d_domain_max"]
    nelems = res[0] - 1

    domain, geom = mesh.line(numpy.linspace(domain_min[0], domain_max[0], nelems+1), periodic=True)
    
    internal_coords = numpy.array([numpy.linspace(domain_min[0], domain_max[0], res[0]), numpy.full(res[0], domain_min[1])]).T
    boundary_coords = numpy.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_max[1]]])
    
    internal_vertex_ids = participant.set_mesh_vertices(mesh_internal_name, internal_coords)
    boundary_vertex_ids = participant.set_mesh_vertices(mesh_boundaries_name, boundary_coords)
    
    sample = domain.locate(geom, internal_coords[:, 0], tol=1e-5)
    internal_coord_to_index = {tuple(coord): i for i, coord in enumerate(internal_coords)}
    boundary_indices_in_internal_mesh = [internal_coord_to_index[tuple(coord)] for coord in boundary_coords]

    # --- Nutils PDE Setup ---
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
    ns.uinit = 'exp(-25 (x - 0.5)^2)'

    res_pde = domain.integral('(v du / dt - ∇(v) f) dV' @ ns, degree=degree*2)
    res_pde -= domain.interfaces.integral('[v] n ({f} - .5 C [u] n) dS' @ ns, degree=degree*2)

    sqr = domain.integral('(u - uinit)^2 dV' @ ns, degree=max(degree*2, 5))
    args = System(sqr, trial='u').solve()
    args['t'] = 0.

    system = System(res_pde, trial='u', test='v')

    if participant.requires_initial_data():
        internal_data_values = sample.eval(ns.u, **args)
        boundary_data_values = internal_data_values[boundary_indices_in_internal_mesh]
        participant.write_data(mesh_internal_name, data_name, internal_vertex_ids, internal_data_values)
        participant.write_data(mesh_boundaries_name, data_name, boundary_vertex_ids, boundary_data_values)

    participant.initialize()

    with log.iter.plain('timestep', itertools.count()) as steps:
        for _ in steps:
            if not participant.is_coupling_ongoing():
                break

            timestep = participant.get_max_time_step_size()

            log.info('time:', round(args['t'], 10))

            args = system.step(timestep=timestep, arguments=args, timearg='t', suffix='0', tol=newtontol)

            internal_data_values = sample.eval(ns.u, **args)
            boundary_data_values = internal_data_values[boundary_indices_in_internal_mesh]
            participant.write_data(mesh_internal_name, data_name, internal_vertex_ids, internal_data_values)
            participant.write_data(mesh_boundaries_name, data_name, boundary_vertex_ids, boundary_data_values)

            participant.advance(timestep)

    participant.finalize()
    return args

if __name__ == '__main__':
    from nutils import cli
    cli.run(main)