

import numpy as np
from scipy.linalg import sqrtm
from quantum.concept_learner import ConceptLearner

from quantum.utils import get_concept_positive_operator, partial_trace_domain


def purity(rho_or_learned_qoncept):
    if isinstance(rho_or_learned_qoncept, ConceptLearner):
        rho = get_concept_positive_operator(rho_or_learned_qoncept)
    else:
        rho = rho_or_learned_qoncept
    rho = rho / np.trace(rho)
    return np.trace(rho @ rho).real

def von_neumann_entropy(rho_or_learned_qoncept):
    if isinstance(rho_or_learned_qoncept, ConceptLearner):
        rho = get_concept_positive_operator(rho_or_learned_qoncept)
    else:
        rho = rho_or_learned_qoncept
    eigvals = np.linalg.eigvals(rho)
    # Drop zero eigenvalues so that log2 is defined
    eigvals = np.array([x for x in eigvals.tolist() if not np.isclose(x, 0)])
    log2_eigvals = np.matrix(np.log2(eigvals))
    eigvals = np.matrix(eigvals)
    S = -np.dot(eigvals, log2_eigvals.H).item().real
    return S

def entanglement_entropy(learned_qoncept):
    rho = get_concept_positive_operator(learned_qoncept)
    domains_to_discard = list(range(len(learned_qoncept.concept_domains[1:])))
    rho = partial_trace_domain(rho, learned_qoncept, domains_to_discard)
    entropy = von_neumann_entropy(rho)
    return entropy

def conditional_entropy(learned_qoncept):
    return von_neumann_entropy(learned_qoncept) - entanglement_entropy(learned_qoncept)

def log_negativity(rho_or_learned_qoncept):
    if isinstance(rho_or_learned_qoncept, ConceptLearner):
        rho = get_concept_positive_operator(rho_or_learned_qoncept)
    else:
        rho = rho_or_learned_qoncept
    rho = np.matrix(rho)
    return np.trace(sqrtm(rho.H @ rho)).real
