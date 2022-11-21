from quantum.measures import purity, von_neumann_entropy, entanglement_entropy, conditional_entropy, log_negativity
from quantum.utils import load_learned_concept, load_saved_model

base_path = '/home/sclark/qoncepts/qoncepts-expts-paper/concepts-vae/saved_models-in-report/'
IMAGE_DIR = '/home/sclark/concept_vae/final_data/train'
QONCEPTS_MODEL = base_path + 'qoncepts_November_14_10_29_fig3'
CONCEPT_DOMAINS = [0, 2] # 0 for colour, 2 for shape
NUM_CONCEPT_PQC_LAYERS = 2
MIXED = False
learned_qoncept_file = base_path + '/learned_concept_November_18_15_56_sec7.4'

qoncepts = load_saved_model(QONCEPTS_MODEL, image_dir=IMAGE_DIR)
learned_qoncept = load_learned_concept(
    learned_qoncept_file,
    qoncepts=qoncepts,
    concept_domains=CONCEPT_DOMAINS,
    num_concept_pqc_layers=NUM_CONCEPT_PQC_LAYERS,
    mixed=MIXED,
    image_dir=IMAGE_DIR,
)

print('purity: {:.4f}'.format(purity(learned_qoncept)))
print('von_neumann_entropy: {:.4f}'.format(von_neumann_entropy(learned_qoncept)))
print('entanglement_entropy: {:.4f}'.format(entanglement_entropy(learned_qoncept)))
if learned_qoncept.mixed:
    print('conditional_entropy: {:.4f}'.format(conditional_entropy(learned_qoncept)))
    print('log_negativity: {:.4f}'.format(log_negativity(learned_qoncept)))
