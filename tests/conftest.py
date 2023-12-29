#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import ase
import numpy as np

from nomad.utils import get_logger
from nomad.units import ureg
from nomad.datamodel import EntryArchive, EntryMetadata, ArchiveSection
from nomad.normalizing import normalizers
from nomad.parsing.parsers import TemplateParser, ArchiveParser
from nomad.normalizing import normalizers
from electronicparsers.vasp import VASPParser
from electronicparsers.exciting import ExcitingParser
from electronicparsers.fhiaims import FHIAimsParser
from electronicparsers.cp2k import CP2KParser
from electronicparsers.crystal import CrystalParser
from electronicparsers.cpmd import CPMDParser
from electronicparsers.nwchem import NWChemParser
from electronicparsers.bigdft import BigDFTParser
from electronicparsers.wien2k import Wien2kParser
from electronicparsers.ams import AMSParser
from electronicparsers.gaussian import GaussianParser
from electronicparsers.abinit import AbinitParser
from electronicparsers.quantumespresso import QuantumEspressoParser
from electronicparsers.orca import OrcaParser
from electronicparsers.castep import CastepParser
from electronicparsers.octopus import OctopusParser
from electronicparsers.gpaw import GPAWParser
from electronicparsers.atk import ATKParser
from electronicparsers.siesta import SiestaParser
from electronicparsers.elk import ElkParser
from electronicparsers.turbomole import TurbomoleParser
from electronicparsers.gamess import GamessParser
from electronicparsers.dmol3 import Dmol3Parser
from electronicparsers.fleur import FleurParser
from electronicparsers.molcas import MolcasParser
from electronicparsers.qbox import QboxParser
from electronicparsers.onetep import OnetepParser
from electronicparsers.psi4 import Psi4Parser
from electronicparsers.yambo import YamboParser
from atomisticparsers.dlpoly import DLPolyParser
from atomisticparsers.libatoms import LibAtomsParser
from atomisticparsers.gulp import GulpParser
from workflowparsers.fhivibes import FHIVibesParser
from workflowparsers.phonopy import PhonopyParser
from workflowparsers.elastic import ElasticParser
from workflowparsers.lobster import LobsterParser
from workflowparsers.aflow import AFLOWParser
from workflowparsers.atomate import AtomateParser
from workflowparsers.asr import ASRParser
from eelsdbparser.eelsdb_parser import EELSDBParser
from nomad.parsing.nexus import NexusParser
import runschema
import simulationworkflowschema


LOGGER = get_logger(__name__)


parsers = {
    'parsers/template': TemplateParser,
    'parsers/vasp': VASPParser,
    'parsers/exciting': ExcitingParser,
    'parsers/fhi-aims': FHIAimsParser,
    'parsers/fhi-vibes': FHIVibesParser,
    'parsers/cp2k': CP2KParser,
    'parsers/crystal': CrystalParser,
    'parsers/cpmd': CPMDParser,
    'parsers/nwchem': NWChemParser,
    'parsers/bigdft': BigDFTParser,
    'parsers/wien2k': Wien2kParser,
    'parsers/ams': AMSParser,
    'parsers/gaussian': GaussianParser,
    'parsers/abinit': AbinitParser,
    'parsers/quantumespresso': QuantumEspressoParser,
    'parsers/orca': OrcaParser,
    'parsers/castep': CastepParser,
    'parsers/dl-poly': DLPolyParser,
    'parsers/lib-atoms': LibAtomsParser,
    'parsers/octopus': OctopusParser,
    'parsers/phonopy': PhonopyParser,
    'parsers/gpaw': GPAWParser,
    'parsers/atk': ATKParser,
    'parsers/gulp': GulpParser,
    'parsers/siesta': SiestaParser,
    'parsers/elk': ElkParser,
    'parsers/elastic': ElasticParser,
    'parsers/turbomole': TurbomoleParser,
    'parsers/gamess': GamessParser,
    'parsers/dmol': Dmol3Parser,
    'parsers/fleur': FleurParser,
    'parsers/molcas': MolcasParser,
    'parsers/qbox': QboxParser,
    'parsers/onetep': OnetepParser,
    'parsers/eels': EELSDBParser,
    'parsers/lobster': LobsterParser,
    'parsers/aflow': AFLOWParser,
    'parsers/atomate': AtomateParser,
    'parsers/asr': ASRParser,
    'parsers/psi4': Psi4Parser,
    'parsers/yambo': YamboParser,
    'parsers/archive': ArchiveParser,
    'parsers/nexus': NexusParser
}


parser_examples = [
    ('parsers/template', 'tests/data/template.json'),
    ('parsers/exciting', 'tests/data/exciting/Ag/INFO.OUT'),
    ('parsers/exciting', 'tests/data/exciting/GW/INFO.OUT'),
    ('parsers/exciting', 'tests/data/exciting/nitrogen/INFO.OUT_nitrogen'),
    ('parsers/exciting', 'tests/data/exciting/nitrogen/INFO.OUT_carbon'),
    ('parsers/vasp', 'tests/data/vasp/vasp.xml'),
    ('parsers/vasp', 'tests/data/vasp_compressed/vasp.xml.gz'),
    ('parsers/vasp', 'tests/data/vasp_outcar/OUTCAR'),
    ('parsers/fhi-aims', 'tests/data/fhi-aims/aims.out'),
    ('parsers/fhi-vibes', 'tests/data/fhi-vibes/molecular_dynamics.nc'),
    ('parsers/cp2k', 'tests/data/cp2k/si_bulk8.out'),
    ('parsers/crystal', 'tests/data/crystal/si.out'),
    ('parsers/cpmd', 'tests/data/cpmd/geo_output.out'),
    ('parsers/nwchem', 'tests/data/nwchem/single_point/output.out'),
    ('parsers/bigdft', 'tests/data/bigdft/n2_output.out'),
    ('parsers/wien2k', 'tests/data/wien2k/AlN/AlN_ZB.scf'),
    ('parsers/ams', 'tests/data/ams/band_adf.out'),
    ('parsers/gaussian', 'tests/data/gaussian/aniline.out'),
    ('parsers/abinit', 'tests/data/abinit/Fe.out'),
    ('parsers/quantumespresso', 'tests/data/quantum-espresso/benchmark.out'),
    ('parsers/orca', 'tests/data/orca/orca3dot2706823.out'),
    ('parsers/castep', 'tests/data/castep/BC2N-Pmm2-Raman.castep'),
    ('parsers/dl-poly', 'tests/data/dl-poly/OUTPUT'),  # timeout on Matid System Classification
    ('parsers/lib-atoms', 'tests/data/lib-atoms/gp.xml'),
    ('parsers/octopus', 'tests/data/octopus/stdout.txt'),
    ('parsers/phonopy', 'tests/data/phonopy/phonopy-FHI-aims-displacement-01/control.in'),
    ('parsers/gpaw', 'tests/data/gpaw/Fe2.gpw'),
    ('parsers/gpaw', 'tests/data/gpaw/H2_lcao.gpw2'),
    ('parsers/atk', 'tests/data/atk/Si2.nc'),
    ('parsers/gulp', 'tests/data/gulp/example6.got'),
    ('parsers/siesta', 'tests/data/siesta/Fe/out'),
    ('parsers/elk', 'tests/data/elk/Al/INFO.OUT'),
    ('parsers/elastic', 'tests/data/elastic/diamond/INFO_ElaStic'),  # 70Mb file 2big4git
    ('parsers/turbomole', 'tests/data/turbomole/acrolein.out'),
    ('parsers/gamess', 'tests/data/gamess/exam01.out'),
    ('parsers/dmol', 'tests/data/dmol3/h2o.outmol'),
    ('parsers/fleur', 'tests/data/fleur/out'),  # double-check
    ('parsers/molcas', 'tests/data/molcas/test000.input.out'),
    ('parsers/qbox', 'tests/data/qbox/01_h2ogs.r'),
    ('parsers/onetep', 'tests/data/onetep/fluor/12-difluoroethane.out'),
    ('parsers/eels', 'tests/data/eels/eels.json'),
    ('parsers/lobster', 'tests/data/lobster/NaCl/lobsterout'),
    ('parsers/aflow', 'tests/data/aflow/Ag1Co1O2_ICSD_246157/aflowlib.json'),
    ('parsers/atomate', 'tests/data/atomate/mp-1/materials.json'),
    ('parsers/asr', 'tests/data/asr/archive_ccdc26c4f32546c5a00ad03a093b73dc.json'),
    ('parsers/psi4', 'tests/data/psi4/adc1/output.ref'),
    ('parsers/yambo', 'tests/data/yambo/hBN/r-10b_1Ry_HF_and_locXC_gw0_em1d_ppa'),
    ('parsers/archive', 'tests/data/archive.json'),
    ('parsers/nexus', 'tests/data/nexus/201805_WSe2_arpes.nxs'),
    ('parsers/nexus', 'tests/data/nexus/SiO2onSi.ellips.nxs'),
]


def parse_file(parser_name:str, filename:str):
    archive = EntryArchive(metadata=EntryMetadata(parser_name=parser_name))
    parsers[parser_name]().parse(filename, archive, LOGGER)
    return archive


def run_normalize(entry_archive: EntryArchive) -> EntryArchive:
    for normalizer_class in normalizers:
        normalizer = normalizer_class(entry_archive)
        normalizer.normalize()
    return entry_archive




def get_template_computation() -> EntryArchive:
    '''Returns a basic archive template for a computational calculation
    '''
    template = EntryArchive()
    run = runschema.run.Run()
    template.run.append(run)
    run.program = runschema.run.Program(name='VASP', version='4.6.35')
    system = runschema.system.System()
    run.system.append(system)
    system.atoms = runschema.system.Atoms(
        lattice_vectors=[
            [5.76372622e-10, 0.0, 0.0],
            [0.0, 5.76372622e-10, 0.0],
            [0.0, 0.0, 4.0755698899999997e-10]
        ],
        positions=[
            [2.88186311e-10, 0.0, 2.0377849449999999e-10],
            [0.0, 2.88186311e-10, 2.0377849449999999e-10],
            [0.0, 0.0, 0.0],
            [2.88186311e-10, 2.88186311e-10, 0.0],
        ],
        labels=['Br', 'K', 'Si', 'Si'],
        periodic=[True, True, True])
    scc = runschema.calculation.Calculation()
    run.calculation.append(scc)
    scc.system_ref = system
    scc.energy = runschema.calculation.Energy(
        free=runschema.calculation.EnergyEntry(value=-1.5936767191492225e-18),
        total=runschema.calculation.EnergyEntry(value=-1.5935696296699573e-18),
        total_t0=runschema.calculation.EnergyEntry(value=-3.2126683561907e-22))
    return template


def get_template_dft() -> EntryArchive:
    '''Returns a basic archive template for a DFT calculation.
    '''
    template = get_template_computation()
    run = template.run[-1]
    method = runschema.method.Method()
    run.method.append(method)
    method.electrons_representation = [runschema.method.BasisSetContainer(
        type='plane waves',
        scope=['wavefunction'],
        basis_set=[runschema.method.BasisSet(
            type='plane waves',
            scope=['valence'],
        )]
    )]
    method.electronic = runschema.method.Electronic(method='DFT')
    xc_functional = runschema.method.XCFunctional(exchange=[runschema.method.Functional(name='GGA_X_PBE')])
    method.dft = runschema.method.DFT(xc_functional=xc_functional)
    scc = run.calculation[-1]
    scc.method_ref = method
    return template


def get_section_system(atoms):
    if runschema:
        system = runschema.system.System()
        system.atoms = runschema.system.Atoms(
            positions=atoms.get_positions() * 1E-10,
            labels=atoms.get_chemical_symbols(),
            lattice_vectors=atoms.get_cell() * 1E-10,
            periodic=atoms.get_pbc())
        return system


def get_template_for_structure(atoms) -> EntryArchive:
    template = get_template_dft()
    template.run[0].calculation[0].system_ref = None
    template.run[0].calculation[0].eigenvalues.append(runschema.calculation.BandEnergies())
    template.run[0].calculation[0].eigenvalues[0].kpoints = [[0, 0, 0]]
    template.run[0].system = None

    # Fill structural information
    # system = template.run[0].m_create(System)
    # system.atom_positions = atoms.get_positions() * 1E-10
    # system.atom_labels = atoms.get_chemical_symbols()
    # system.simulation_cell = atoms.get_cell() * 1E-10
    # system.configuration_periodic_dimensions = atoms.get_pbc()
    system = get_section_system(atoms)
    template.run[0].system.append(system)

    return run_normalize(template)


@pytest.fixture(params=parser_examples, ids=lambda spec: '%s-%s' % spec)
def parsed_example(request) -> EntryArchive:
    parser_name, mainfile = request.param
    result = parse_file(parser_name, mainfile)
    return result


@pytest.fixture
def normalized_example(parsed_example: EntryArchive) -> EntryArchive:
    return run_normalize(parsed_example)


@pytest.fixture(scope='session')
def single_point() -> EntryArchive:
    '''Single point calculation.'''
    template = get_template_dft()
    return run_normalize(template)


@pytest.fixture(scope='session')
def molecular_dynamics() -> EntryArchive:
    '''Molecular dynamics calculation.'''
    template = get_template_dft()
    run = template.run[0]

    # Create calculations
    n_steps = 10
    calcs = []
    for step in range(n_steps):
        system = runschema.system.System()
        run.system.append(system)
        calc = runschema.calculation.Calculation()
        calc.system_ref = system
        calc.time = step
        calc.step = step
        calc.volume = step
        calc.pressure = step
        calc.temperature = step
        calc.energy = runschema.calculation.Energy(
            potential=runschema.calculation.EnergyEntry(value=step),
        )
        rg_values = runschema.calculation.RadiusOfGyrationValues(
            value=step,
            label='MOL',
            atomsgroup_ref=system
        )
        calc.radius_of_gyration = [runschema.calculation.RadiusOfGyration(
            kind='molecular',
            radius_of_gyration_values=[rg_values],
        )]
        calcs.append(calc)
        run.calculation.append(calc)

    # Create workflow
    diff_values = simulationworkflowschema.molecular_dynamics.DiffusionConstantValues(
        value=2.1,
        error_type='Pearson correlation coefficient',
        errors=0.98,
    )
    msd_values = simulationworkflowschema.molecular_dynamics.MeanSquaredDisplacementValues(
        times=[0, 1, 2],
        n_times=3,
        value=[0, 1, 2],
        label='MOL',
        errors=[0, 1, 2],
        diffusion_constant=diff_values,
    )
    msd = simulationworkflowschema.molecular_dynamics.MeanSquaredDisplacement(
        type='molecular',
        direction='xyz',
        error_type='bootstrapping',
        mean_squared_displacement_values=[msd_values],
    )
    rdf_values = simulationworkflowschema.molecular_dynamics.RadialDistributionFunctionValues(
        bins=[0, 1, 2],
        n_bins=3,
        value=[0, 1, 2],
        frame_start=0,
        frame_end=100,
        label='MOL-MOL',
    )
    rdf = simulationworkflowschema.molecular_dynamics.RadialDistributionFunction(
        type='molecular',
        radial_distribution_function_values=[rdf_values],
    )
    results = simulationworkflowschema.molecular_dynamics.MolecularDynamicsResults(
        radial_distribution_functions=[rdf],
        mean_squared_displacements=[msd],
    )
    method = simulationworkflowschema.molecular_dynamics.MolecularDynamicsMethod(
        thermodynamic_ensemble='NVT',
        integration_timestep=0.5 * ureg('fs'),
    )
    md = simulationworkflowschema.molecular_dynamics.MolecularDynamics(
        results=results, method=method
    )
    results.calculation_result_ref = calcs[-1]
    results.calculations_ref = calcs
    template.workflow2 = md

    return run_normalize(template)


@pytest.fixture(scope='session')
def phonon() -> EntryArchive:
    parser_name = 'parsers/phonopy'
    filepath = 'tests/data/phonopy/phonopy-FHI-aims-displacement-01/control.in'
    archive = parse_file(parser_name, filepath)
    return run_normalize(archive)


@pytest.fixture(scope='session')
def geometry_optimization() -> EntryArchive:
    template = get_template_dft()
    template.run[0].system = None
    template.run[0].calculation = None
    run = template.run[0]
    atoms1 = ase.build.bulk('Si', 'diamond', cubic=True, a=5.431)
    atoms2 = ase.build.bulk('Si', 'diamond', cubic=True, a=5.431)
    atoms2.translate([0.01, 0, 0])
    sys1 = get_section_system(atoms1)
    sys2 = get_section_system(atoms2)
    scc1 = runschema.calculation.Calculation()
    scc2 = runschema.calculation.Calculation()
    scc1.energy = runschema.calculation.Energy(
        total=runschema.calculation.EnergyEntry(value=1e-19),
        total_t0=runschema.calculation.EnergyEntry(value=1e-19))
    scc2.energy = runschema.calculation.Energy(
        total=runschema.calculation.EnergyEntry(value=0.5e-19),
        total_t0=runschema.calculation.EnergyEntry(value=0.5e-19))
    scc1.system_ref = sys1
    scc2.system_ref = sys2
    scc1.method_ref = run.method[0]
    scc2.method_ref = run.method[0]
    run.system.append(sys1)
    run.system.append(sys2)
    run.calculation.append(scc1)
    run.calculation.append(scc2)

    template.workflow2 = simulationworkflowschema.GeometryOptimization(
        method=simulationworkflowschema.GeometryOptimizationMethod(
            convergence_tolerance_energy_difference=1e-3 * ureg.electron_volt,
            convergence_tolerance_force_maximum=1e-11 * ureg.newton,
            convergence_tolerance_displacement_maximum=1e-3 * ureg.angstrom,
            method='bfgs',
            type='atomic'))
    template.workflow2.normalize(template, get_logger(__name__))

    run_normalize(template)
    return template


@pytest.fixture(scope='session')
def bulk() -> EntryArchive:
    atoms = ase.build.bulk('Si', 'diamond', cubic=True, a=5.431)
    return get_template_for_structure(atoms)


@pytest.fixture(scope='session')
def atom() -> EntryArchive:
    atoms = ase.Atoms(
        symbols=['H'],
        scaled_positions=[[0.5, 0.5, 0.5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    return get_template_for_structure(atoms)


@pytest.fixture(scope='session')
def molecule() -> EntryArchive:
    atoms = ase.build.molecule('CO2')
    return get_template_for_structure(atoms)


@pytest.fixture(scope='session')
def one_d() -> EntryArchive:
    atoms = ase.build.graphene_nanoribbon(1, 1, type='zigzag', vacuum=10, saturated=True)
    return get_template_for_structure(atoms)


@pytest.fixture(scope='session')
def two_d() -> EntryArchive:
    atoms = ase.Atoms(
        symbols=['C', 'C'],
        scaled_positions=[
            [0, 0, 0.5],
            [1 / 3, 1 / 3, 0.5],
        ],
        cell=[
            [2.461, 0, 0],
            [np.cos(np.pi / 3) * 2.461, np.sin(np.pi / 3) * 2.461, 0],
            [0, 0, 20]
        ],
        pbc=True
    )
    return get_template_for_structure(atoms)


@pytest.fixture(scope='session')
def surface() -> EntryArchive:
    atoms = ase.build.fcc111('Al', size=(2, 2, 3), vacuum=10.0)
    return get_template_for_structure(atoms)


