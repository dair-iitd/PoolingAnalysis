from numpy import linalg as LA
import numpy as np
import ipdb

def process_gradients(grad_norms):
    # num_words * 4gate_sizes * hidden_dim
    num_words = len(grad_norms)
    xp = np.linspace(0,99,num_words)
    fp = [g.item() for g in grad_norms]
    x = list(range(100))
    interp_grads = np.interp(x, xp, fp)
    return interp_grads

def compute_norm(grads):
    return LA.norm(grads)

def gates_compute(model, args, gates_f):
    ih_gate = model.rnn.forward_cell.weight_ih.tolist()
    ih_bias = model.rnn.forward_cell.bias_ih.tolist()
    hh_gate = model.rnn.forward_cell.weight_hh.tolist()
    hh_bias = model.rnn.forward_cell.bias_hh.tolist()
    ii, if_, ig, io = compute_norm(ih_bias[:HIDDEN_DIM]), compute_norm(ih_bias[HIDDEN_DIM:2*HIDDEN_DIM]), compute_norm(ih_bias[2*HIDDEN_DIM:3*HIDDEN_DIM]), compute_norm(ih_bias[3*HIDDEN_DIM:])
    hi, hf_, hg, ho = compute_norm(hh_bias[:HIDDEN_DIM]), compute_norm(hh_bias[HIDDEN_DIM:2*HIDDEN_DIM]), compute_norm(hh_bias[2*HIDDEN_DIM:3*HIDDEN_DIM]), compute_norm(hh_bias[3*HIDDEN_DIM:])
    
    inorm, fnorm, gnorm, onorm = compute_norm([ii, hi]), compute_norm([if_, hf_]), compute_norm([ig, hg]), compute_norm([io, ho])
    gates_f.write(str(inorm)+'\t'+str(fnorm)+'\t'+str(gnorm)+'\t'+str(onorm)+'\n')
    return

def gradients_compute(model, args, all_gradients):
    ht_norms = model.rnn.concatenate_lists(model.rnn.f_ht, model.rnn.b_ht)
    ht_norms = ht_norms[::-1]
    ht_grads = process_gradients(ht_norms)
    all_gradients.append(ht_grads)

    return 

def write_gradients(fp, args, all_gradients, model_dir):
    sum_gradients = (sum(all_gradients) / len(all_gradients)).tolist()
    f = open(model_dir + '/' + fp, 'w')
    f.write('\n'.join([str(g) for g in sum_gradients]))
    f.close()
    return


def compute_ratios(all_gradients):
    gradients = np.mean(all_gradients[-10:], axis=0)
    first_ratio = gradients[0] / np.average(gradients[45:55])
    sum_ratio = (gradients[0]+gradients[-1]) / np.average(gradients[45:55])
    return first_ratio, sum_ratio

def free_stored_grads(model):
    model.rnn.f_hh, model.rnn.f_ih, model.rnn.f_ht = [], [], []
    model.rnn.b_hh, model.rnn.b_ih, model.rnn.b_ht = [], [], []
    model.rnn.forward_acts, model.rnn.backward_acts = [], []
    model.rnn.grad_fi, model.rnn.grad_ff, model.rnn.grad_fo = [], [], []
    model.rnn.grad_bi, model.rnn.grad_bf, model.rnn.grad_bo = [], [], []
    return



            
