import torch
import yaml 
import shap
import numpy as np
from utils.init_model import init_model
from utils.load_data import load_dataset
from argparse import Namespace
import matplotlib.pyplot as plt

# eye_in_walkingree, expt: 20250908_003251_AuxLSTM
# eye_in_walkingree, expt + person2: 20250908_182530_AuxLSTM

# new: 20250919_053434_LSTMse

data_folder = 'data/indiv_time_o40_p40_s4'
ckpt_folder = 'logs/indiv_time_o40_p40_s4/sqlite_final/20250919_053434_LSTMse'
filename = ckpt_folder.split('/')[-1]
ckpt_path = f'{ckpt_folder}/checkpoints/best_model.pth'
config_path = f'{ckpt_folder}/config.yaml'
device = torch.device('cpu')

with open(config_path, 'r') as f:
    yaml_args = yaml.safe_load(f)
yaml_args['model'] = 'multimodallstm'
yaml_args = Namespace(**yaml_args)

# load model
model = init_model(yaml_args, device=device)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval() 

# load data 
data = load_dataset(data_folder, base_motion=yaml_args.base_motion, target=yaml_args.target,
             use_headeye=yaml_args.use_headeye, use_pod=yaml_args.use_pod,
             use_expt=yaml_args.use_expt, use_person=yaml_args.use_person, aux_format=yaml_args.aux_format)
test_x, test_y, test_aux = data['x_test'], data['y_test'], data['aux_test']
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
test_aux = torch.tensor(test_aux, dtype=torch.float32).to(device)

feature_names = []
if yaml_args.use_expt: 
    feature_names.extend(['eHMI presence', 'shuttle behavior', 'approaching angle', 'the number of shuttles'])
if yaml_args.use_person is not None:
    if 'age' in yaml_args.use_person: 
        feature_names.append('age')
    if 'gender'in yaml_args.use_person: 
        feature_names.append('gender')
    if 'as' in yaml_args.use_person:
        feature_names.extend(['as_familiarity', 'as_experience'])
    if 'pb' in yaml_args.use_person:
        feature_names.extend(['v_score', 'l_score', 'p_score'])
    if 'trust' in yaml_args.use_person:
        feature_names.append('trust_score')
print(feature_names)

sample_idx = np.random.randint(0, len(test_x), size=100)

# shap value 
explainer = shap.GradientExplainer(model, [test_x[sample_idx], test_aux[sample_idx]])
shap_values = explainer.shap_values([test_x[sample_idx][:50], test_aux[sample_idx][:50]])

shap.summary_plot(shap_values=shap_values[1][:, :, 0], 
                  features=test_aux[sample_idx][:50], 
                  feature_names=feature_names, show=False)
plt.title("x")
plt.savefig(f'figures/{filename}_shap_summary_aux_on_x.png', bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values=shap_values[1][:, :, 1], 
                  features=test_aux[sample_idx][:50], 
                  feature_names=feature_names, show=False)
plt.title("y")
plt.savefig(f'figures/{filename}_shap_summary_aux_on_y.png', bbox_inches='tight')
plt.close()