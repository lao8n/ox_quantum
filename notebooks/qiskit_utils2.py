"""
    A selection of Qiskit utility functions used during the course.
"""
# pylint: disable = unused-import

import sys
if sys.version_info[:2] >= (3,9):
    from collections.abc import Iterable, Mapping, Sequence, Sized
else:
    from typing import Iterable, Mapping, Sequence, Sized
from itertools import product
from math import pi
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypedDict, TypeVar, Union

import numpy as np
import networkx as nx # type: ignore

import qiskit # type: ignore
from qiskit import QuantumCircuit, transpile, execute, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend # type: ignore
from qiskit.providers.aer import AerSimulator # type: ignore
from qiskit.circuit import Parameter, Qubit, Clbit # type: ignore
from qiskit.visualization import (plot_bloch_multivector, plot_state_paulivec, # type: ignore
                                  plot_histogram, plot_bloch_vector)
from qiskit.quantum_info import Statevector, DensityMatrix # type: ignore
from qiskit.result import marginal_counts # type: ignore
from qiskit.visualization.bloch import Bloch # type: ignore
from qiskit.visualization.utils import _bloch_multivector_data # type: ignore

if sys.version_info[:2] >= (3,9):

    FloatArray = np.ndarray[float, np.dtype[np.floating[Any]]]
    """ Type for a float Numpy array. """

    ComplexArray = np.ndarray[complex, np.dtype[np.floating[Any]]]
    """ Type for a complex Numpy array. """

else:

    FloatArray = np.ndarray
    """ Type for a float Numpy array. """

    ComplexArray = np.ndarray
    """ Type for a complex Numpy array. """


State = Union[ComplexArray, QuantumCircuit, Statevector, DensityMatrix]
"""
    Type alias for the states accepted by `bloch_multivector_data`
"""

def bloch_multivector_data(state: State) -> Sequence[Sequence[FloatArray]]:
    """
        Return list of Bloch vectors for each qubit.
        See `qiskit.visualization.utils._bloch_multivector_data`.
    """
    return _bloch_multivector_data(state) # type: ignore

def cx_tree(num_qubits: int, *, rev: bool = False) -> QuantumCircuit:
    """
        Builds a CX tree on `num_qubits` (leaves first), for use in phase gadgets and Pauli gadgets.
        The optional `rev` kwarg can be used to build the tree in reverse (root first)
    """
    n = num_qubits
    circ = QuantumCircuit(n)
    if num_qubits <= 1:
        return circ
    qubits = range(n)
    p = n//2+n%2
    q = n-p
    if rev:
        # if `rev`, the root CX gate is applied before recursing:
        circ.cx(qubits[p-1], qubits[n-1])
    if n > 2:
        # if n >= 3, CX sub-trees are built recursively and composed into the circuit:
        circ.compose(cx_tree(p, rev=rev), qubits=qubits[:p], inplace=True)
        circ.compose(cx_tree(q, rev=rev), qubits=qubits[p:], inplace=True)
    if not rev:
        # if not `rev`, the root CX gate is applied after recursing:
        circ.cx(qubits[p-1], qubits[n-1])
    return circ

def z_phase_gadget(num_qubits: int, angle: Union[float, Parameter]) -> QuantumCircuit:
    """
        Builds a phase gadgets on `num_qubits`, with the given angle.
    """
    assert num_qubits >= 1
    circ = QuantumCircuit(num_qubits)
    # CX tree -> RZ(angle) -> reversed CX tree
    circ.compose(cx_tree(num_qubits), inplace=True)
    circ.rz(angle, num_qubits-1)
    circ.compose(cx_tree(num_qubits, rev=True), inplace=True)
    return circ

basis_change_gate = {"X": "h", "Y": "sx", "Z": "i"}
""" Entry `B: g` is the gate to change from basis `B` to Z basis. """

basis_change_gate_dg = {"X": "h", "Y": "sxdg", "Z": "i"}
""" Entry `B: g` is the gate to change from Z basis to basis `B`. """

def pauli_gadget(paulis: str, angle: Union[float, Parameter],
                 *, r2l: bool = True) -> QuantumCircuit:
    """
        Builds a Pauli gadget, with the given Paulis and angle.
        The number of qubits is the length of `paulis`.
        The optional `r2l` kwarg determines whether the Paulistring is to be read right-to-left
        (default, consistend with Qiskit's own ordering) or left-to-right.
    """
    assert len(paulis) >= 1
    assert all(p in ("X", "Y", "Z", "I") for p in paulis)
    n = len(paulis)
    qubits = range(n)
    circ = QuantumCircuit(n)
    if r2l:
        # If paulistring is to be read right-to-left, invert it:
        paulis = paulis[::-1]
    legs = [q for q in qubits if paulis[q] != "I"]
    m = len(legs)
    for p, q in zip(paulis, qubits):
        if p not in ("I", "Z"):
            # change basis for qubit `q` from `paulis[q]` to Z basis:
            g = basis_change_gate[p]
            getattr(circ, g)(q) # this is `circ.h(q)`, `circ.sx(q)` or `circ.i(q)`
    # Compose a phase gadget into the circuit:
    circ.compose(z_phase_gadget(m, angle), qubits=legs, inplace=True)
    for p, q in zip(paulis, qubits):
        if p not in ("I", "Z"):
            # change basis for qubit `q` from Z basis back to `paulis[q]`:
            g = basis_change_gate_dg[p]
            getattr(circ, g)(q) # this is `circ.h(q)`, `circ.sxdg(q)` or `circ.i(q)`
    return circ

def pauli_meas(paulis: str, *, r2l: bool = True) -> QuantumCircuit:
    """
        Applies a Pauli measurement to each qubit, specified by a given Paulistring,
        with I indicating that the qubits should not be measured.
        The number of qubits is the length of `paulis`.
        The optional `r2l` kwarg determines whether the Paulistring is to be read right-to-left
        (default, consistend with Qiskit's own ordering) or left-to-right.
    """
    assert all(p in ("X", "Y", "Z", "I") for p in paulis)
    assert len(paulis) >= 1
    if r2l:
        # If paulistring is to be read right-to-left, invert it:
        paulis = paulis[::-1]
    n = len(paulis)
    # only qubits `q` with `paulis[q] != "I"` are measured:
    measured_qubits = [q for q in range(n) if paulis[q] != "I"]
    circ = QuantumCircuit(n, n)
    for q in measured_qubits:
        # for measured qubits, change basis from `paulis[q]` to Z basis:
        g = basis_change_gate[paulis[q]]
        getattr(circ, g)(q) # this is `circ.h(q)`, `circ.sx(q)` or `circ.i(q)`
    # measure qubits `q` in Z basis, store outcome at position `q` of classical reg:
    circ.measure(measured_qubits, measured_qubits)
    return circ

def paulistrs(n: int, *, include_id: bool = True) -> Sequence[str]:
    """
        Returns the sequence of all Paulistrings on `n` qubits.
        The optional `include_id` kwarg determines whether the I matrix
        should be included (default) or not.
    """
    ps = (["I"] if include_id else [])+["X", "Y", "Z"]
    return tuple("".join(t)
                 for t in product(ps, repeat=n)
                 if not all(p == "I" for p in t))

def expval(counts: Mapping[str, float]) -> float:
    """
        Computes the expectation value from a counts dictionary.
    """
    shots = sum(counts.values())
    p0 = sum(c/shots for b, c in counts.items() # probability = counts/shots
             if b.count("1")%2 == 0) # even bitsum
    p1 = sum(c/shots for b, c in counts.items()
             if b.count("1")%2 == 1) # odd bitsum
    return p0-p1 # expval = provavility of even bitsum - probability of odd bitsum

def make_params(fst: int, snd: Optional[int] = None, label: str = "θ") -> Sequence[Parameter]:
    """
        Makes a list of parameters labelled with the given range of indices, e.g. `"θ[2]"`:

        - `make_params(n)` returns `[Parameter("θ[0]"),...,Parameter(f"θ[{n-1}]")]`
        - `make_params(start, end)` returns `[Parameter("θ[{start}]"),...,Parameter(f"θ[{end-1}]")]`

        The optional kwarg `label` can be used to specify an alternative label for the parameters,
        e.g. `make_params(n, 't')` returns `[Parameter("t[0]"),...,Parameter(f"t[{n-1}]")]`
    """
    start = 0 if snd is None else fst
    end = fst if snd is None else snd
    assert end >= start
    return tuple(Parameter(f"{label}[{idx}]") for idx in range(start, end))

_T_co = TypeVar("_T_co", covariant=True)
""" A covariant type variable, for use by `SizedIterable` below. """

class SizedIterable(Sized, Iterable[_T_co], Protocol):
    """ Protocol (structural type) for iterables which have a size. """
    # pylint: disable = abstract-method, too-few-public-methods
    ...

class ObjFun(Protocol):
    """ Protocol (structural type) for objective functions. """
    # pylint: disable = abstract-method, too-few-public-methods

    def __call__(self, param_vals: SizedIterable[float]) -> float:
        ...

class ObjFunHist(TypedDict, total=True):
    """
        Typed dictionary for the history of objective function values and parameter values.
    """
    param_vals: List[SizedIterable[float]]
    obj_val: List[float]

def gradient(obj_fun: ObjFun,
             param_vals: SizedIterable[float]) -> FloatArray:
    """
        Computes the gradient for the given objective function at the given parameter values.
        For `m` parameters, makes `2*m` calls to `obj_fun`.
        Returns the gradient as an `m`-dimensional vector.
    """
    m = len(param_vals)
    # generate all parameter shifts:
    shifted_param_vals = []
    for j in range(m):
        # create the basis vector $e_j$:
        ej = np.zeros(m)
        ej[j] = 1
        # append param shifts for parameter j:
        shifted_param_vals.append(param_vals + pi/2*ej)
        shifted_param_vals.append(param_vals - pi/2*ej)
    # evaluate all parameters shifts:
    obj_fun_vals = [
        obj_fun(param_vals)
        for param_vals in shifted_param_vals
    ]
    # compute gradient from param shift objective values:
    grad = np.zeros(m)
    for j in range(m):
        grad[j] = (obj_fun_vals[2*j]-obj_fun_vals[2*j+1])/2
    return grad

class GradFun(Protocol):
    """ Protocol (structural type) for gradient functions. """
    # pylint: disable = abstract-method, too-few-public-methods

    def __call__(self, param_vals: SizedIterable[float]) -> FloatArray:
        ...

class GradFunHist(TypedDict, total=True):
    """
        Typed dictionary for the history of gradient values, parameter values
        and objective function values returned by `make_gradient_function`.
    """
    grad: List[SizedIterable[float]]
    param_vals: List[SizedIterable[float]]
    obj_val: List[float]

def make_grad_fun(obj_fun: ObjFun, store_obj_val: bool = True) -> Tuple[GradFun, GradFunHist]:
    """
        Makes a gradient function from a given objective function.
        Returns `grad_fun, hist`, where `grad_fun` is a gradient function (for use in Qiskit
        optimizers) and `hist` is a dictionary containing the history of gradient values,
        parameter values, and optionally objective function values.
        The latter are computed only if `store_obj_val is True`.
    """
    hist: GradFunHist = {
        "grad": [],
        "param_vals": [],
        "obj_val": []
    }
    def grad_fun(param_vals: SizedIterable[float]) -> FloatArray:
        grad = gradient(obj_fun, param_vals)
        hist["grad"].append(grad)
        hist["param_vals"].append(param_vals)
        if store_obj_val:
            obj_val = obj_fun(param_vals)
            hist["obj_val"].append(obj_val)
        return grad
    return grad_fun, hist

def obs_expval(state_prep: QuantumCircuit, obs: Mapping[str, float],
               *, r2l: bool = True, **kwargs: Any) -> float:
    """
        Computes the expectation value of the given observable `obs`
        on the state prepared by the given `state_prep` circuit.
        The optional `r2l` kwarg is passed to `pauli_meas`,
        while the remaining `kwargs` are passed to `execute`.
    """
    # pylint: disable = too-many-locals
    circ = state_prep.copy()
    n = state_prep.num_qubits
    assert all(len(paulis) == n for paulis in obs)
    # compute list of required Pauli measurements:
    required_paulis_list = sorted({paulis.replace("I", "Z") for paulis in obs
                                   if not all(p == "I" for p in paulis)})
    # perform tomography of required Pauli measurements:
    circ.add_register(ClassicalRegister(n))
    tomography_circuits = {paulis: circ.compose(pauli_meas(paulis, r2l=r2l))
                           for paulis in required_paulis_list}
    batch = list(tomography_circuits.values())
    job = execute(batch, **kwargs)
    counts_list = job.result().get_counts()
    if not isinstance(counts_list, list):
        counts_list = [counts_list]
    tomography_counts = dict(zip(tomography_circuits.keys(), counts_list))
    # Fill in remaining Pauli measurements by marginalisation:
    for paulis in set(obs)-set(tomography_counts)-{"I"*n}:
        qubits_to_measure = [q for q in range(n) if paulis[n-1-q] != "I"]
        full_counts = tomography_counts[paulis.replace("I", "Z")]
        tomography_counts[paulis] = marginal_counts(full_counts, qubits_to_measure)
    # Compute expectation values for Pauli observables:
    expvals = {"I"*n: 1.0} | {
        paulis: expval(counts)
        for paulis, counts in tomography_counts.items()
    }
    # Compute and return observable expectation value:
    return sum(obs_val*expvals[paulis] for paulis, obs_val in obs.items())


def energy(H_prob: Mapping[str, float], bitstr: str) -> float:
    """
        Returns the energy associated by a given problem Hamiltonian to a bitstring.
    """
    n = len(list(H_prob)[0])
    assert all(len(paulis) == n for paulis in H_prob)
    assert all(all(p in {"I", "Z"} for p in paulis) for paulis in H_prob)
    assert len(bitstr) == n
    assert all(b in {"0", "1"} for b in bitstr)
    e = 0.0
    for paulis, coeff in H_prob.items():
        sign = (-1)**len([q for q in range(n) if paulis[q] == "Z" and bitstr[q] == "1"])
        e += sign*coeff
    return e

def H_prob_validate(H: Mapping[str, float]) -> int:
    """
        Validates a problem Hamiltonian, returning the number of qubits.
    """
    n = len(list(H)[0])
    assert all(len(paulis) == n for paulis in H)
    assert all(all(p in {"I", "Z"} for p in paulis) for paulis in H)
    return n

def H_sumprod(*Hs: Mapping[str, float],
              times: Union[int, float, Sequence[Union[int, float]]] = 1.0) -> Mapping[str, float]:
    """
        Sums the given problem Hamiltonians, optionally multiplying them
        by a scalar (or a sequence of scalars).
    """
    if isinstance(times, (int, float)):
        times = [times]*len(Hs)
    if not Hs:
        return {}
    n = H_prob_validate(Hs[0])
    assert all(H_prob_validate(H) == n for H in Hs[1:])
    H_sum: Dict[str, float] = {}
    for i, H in enumerate(Hs):
        for paulis, coeff in H.items():
            H_sum[paulis] = H_sum.get(paulis, 0.0)+coeff*times[i]
    return H_sum

def c_bit(i: int, n: int) -> Mapping[str, float]:
    """
        Problem Hamiltonian for a single bit value $b_i$ (out of bits $b_0,...,b_{n-1}$).
    """
    return {"I"*(n-1-i)+"Z"+"I"*i: 1.0}

def c_neg(H_phi: Mapping[str, float]) -> Mapping[str, float]:
    """
        Logical NOT of a given problem Hamiltonian.
    """
    H_prob_validate(H_phi)
    return {paulis: -coeff for paulis, coeff in H_phi.items()}

def c_xor(H_phi: Mapping[str, float],
          H_psi: Mapping[str, float]) -> Mapping[str, float]:
    """
        Logical XOR of two given problem Hamiltonians.
    """
    n = H_prob_validate(H_phi)
    assert H_prob_validate(H_psi) == n
    H = {}
    for (p_phi, c_phi), (p_psi, c_psi) in product(H_phi.items(), H_psi.items()):
        paulis = ''.join("I" if p_phi[q] == p_psi[q] else "Z" for q in range(n))
        H[paulis] = c_phi*c_psi
    return H

def c_and(H_phi: Mapping[str, float],
          H_psi: Mapping[str, float]) -> Mapping[str, float]:
    """
        Logical AND of two given problem Hamiltonians.
    """
    H_nxor = c_neg(c_xor(H_phi, H_psi))
    n = len(list(H_nxor)[0])
    return H_sumprod(H_nxor, H_phi, H_psi, {"I"*n: 1}, times=0.5)

def c_or(H_phi: Mapping[str, float],
         H_psi: Mapping[str, float]) -> Mapping[str, float]:
    """
        Logical OR of two given problem Hamiltonians.
    """
    H_xor = c_xor(H_phi, H_psi)
    n = len(list(H_xor)[0])
    return H_sumprod(H_xor, H_phi, H_psi, {"I"*n: -1}, times=0.5)

def c_eq(H_phi: Mapping[str, float],
         H_psi: Union[Mapping[str, float], str]) -> Mapping[str, float]:
    """
        Logical equality of two given problem Hamiltonians.
    """
    if isinstance(H_psi, str):
        assert H_psi in {"0", "1"}
        return H_phi if H_psi == "1" else c_neg(H_phi)
    return c_neg(c_xor(H_phi, H_psi))

def H_maxcut(G: nx.Graph) -> Mapping[str, float]:
    """
        Problem Hamiltonian for the MAX-CUT problem on a given graph.
    """
    n = len(G.nodes)
    edge_constraints = [c_xor(c_bit(i, n), c_bit(j, n)) for i, j in G.edges]
    return H_sumprod(*edge_constraints)

def H_ising_validate(H: Mapping[str, float]) -> int:
    """
        Validates a Ising Hamiltonian, returning the number of qubits.
    """
    n = H_prob_validate(H)
    assert all(paulis.count("Z") <= 2 for paulis in H)
    return n

def draw_ising(H: Mapping[str, float], **kwargs: Any) -> None:
    """
        Draws an Ising Hamiltonian using NetworkX.
    """
    n = H_ising_validate(H)
    _H = {tuple(q for q, p in enumerate(paulis) if p == "Z"): coeff
                 for paulis, coeff in H.items()}
    biases = {paulis: coeff for paulis, coeff in _H.items() if len(paulis) == 1}
    couplings = {paulis: coeff for paulis, coeff in _H.items() if len(paulis) == 2}
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    G.add_edges_from(list(couplings))
    pos = nx.planar_layout(G)
    node_labels = {q: f"{biases.get((q,), 0.0):.1f}" for q in range(n)}
    edge_labels = {e: f"{coeff:.1f}" for e, coeff in couplings.items()}
    nx.draw_networkx(G, pos, labels=node_labels, node_color = "#dddddd", **kwargs)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

def trotterize(H: Mapping[str, Union[float, Callable[[float], float]]],
               K: int, *, t: float = 1.0, barriers: bool = True) -> QuantumCircuit:
    """
        Trotterizes the given Hamiltonian H into K steps.
        By default, the evolution is evaluated at `t=1.0`,
        but other values of `t` can be specified.
    """
    n = len(list(H)[0])
    assert all(len(paulis) == n for paulis in H)
    circ = QuantumCircuit(n)
    for k in range(K):
        dt_k = t/K
        t_k = (k+0.5)*dt_k
        for paulis, coeff in H.items():
            if not isinstance(coeff, float):
                coeff = coeff(t_k)
            circ.compose(pauli_gadget(paulis, 2*coeff*dt_k), inplace=True)
        if barriers:
            circ.barrier()
    return circ

def H_qaa(H_prob: Mapping[str, float], *,
          a: Callable[[float], float],
          b: Callable[[float], float]) -> Mapping[str, Union[float,Callable[[float], float]]]:
    """
        Hamiltonian for the quantum adiabatic algorithm (QAA), for given problem Hamiltonian
        and annealing schedule functions `a` and `b`.
    """
    n = len(list(H_prob)[0])
    assert all(all(p in {"I", "Z"} for p in paulis) for paulis in H_prob)
    H_tran = {("I"*(n-1-i)+"X"+"I"*i):-1.0 for i in range(n)}
    return {
        **{paulis: lambda t: coeff*b(t) for paulis, coeff in H_prob.items()},
        **{paulis: lambda t: coeff*a(t) for paulis, coeff in H_tran.items()}
    }

def qaa_maxcut(G: nx.Graph, K: int, *, t: float = 1,
               a: Callable[[float], float], b: Callable[[float], float],
               barriers: bool = True) -> QuantumCircuit:
    """
        Circuit applying the quantum adiabatic algorithm (QAA)
        for the MAX-CUT problem on given graph.
    """
    n = len(G.nodes)
    H = H_qaa(H_maxcut(G), a=a, b=b)
    circ = QuantumCircuit(n)
    circ.h(range(n)) # init state |+...+>
    if barriers:
        circ.barrier()
    circ.compose(trotterize(H, K, t=t, barriers=barriers), inplace=True)
    return circ

def qaoa_ansatz(H_prob: Mapping[str, float], K: int,
                *, barriers: bool = True) -> QuantumCircuit:
    """
        QAOA ansatz for a given problem Hamiltonian and number of trotterization steps.
    """
    n = H_prob_validate(H_prob)
    alphas = make_params(K, label="α")
    gammas = make_params(K, label="γ")
    circ = QuantumCircuit(n)
    circ.h(range(n)) # init state |+...+>
    for k in range(K):
        alpha_k = alphas[k]
        gamma_k = gammas[k]
        for paulis, coeff in H_prob.items():
            circ.compose(pauli_gadget(paulis, 2*gamma_k*coeff), inplace=True)
        if barriers:
            circ.barrier()
        circ.rx(-2*alpha_k, range(n)) # note the negative sign!
        if barriers:
            circ.barrier()
    return circ

def make_qaoa_obj_fun(H_prob: Mapping[str, float], K: int,
                      *, backend: Backend, **kwargs: Any) -> Tuple[ObjFun, ObjFunHist]:
    """
        Returns the QAOA objective function (and container for objective values and param value)
        for the given problem Hamiltonian and number of trotterization steps.
    """
    hist: ObjFunHist = {"obj_val": [], "param_vals": []}
    ansatz = qaoa_ansatz(H_prob, K)
    def obj_fun(param_vals: SizedIterable[float]) -> float:
        assert len(param_vals) == 2*K
        circ = ansatz.assign_parameters(dict(zip(ansatz.parameters, param_vals)))
        obj_val = obs_expval(circ, H_prob, backend=backend, **kwargs)
        hist["obj_val"].append(obj_val)
        hist["param_vals"].append(param_vals)
        return obj_val
    return obj_fun, hist
